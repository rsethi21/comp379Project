import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.utils import shuffle
import pdb
from gridsearch import grid_search_multi


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input path to diabetes file", required=True)
parser.add_argument("-p", "--predictor", help="name of prediction column", required=True)
parser.add_argument("-s", "--scale", help="columns to scale", nargs="*", required=False, default=[])

class Dataset:
    def __init__(self, filepath, predictor, scale_columns, split=0.2, val_split=None, rs = 4):
        self.rs = rs
        self.open_dataset(filepath)
        self.scale_data(scale_columns)
        self.separate_samples(self.data, predictor, split, val_split)

    def open_dataset(self, filepath):
        self.data = pd.read_csv(filepath, header = 0)

    def separate_samples(self, data, predictor, split, val_split=None):
        self.columns = data.columns
        data = data.dropna()
        data = data.sample(frac=1, random_state=self.rs)
        y = data.loc[:,predictor].to_numpy()
        X = data.drop(columns=[predictor]).to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=self.rs, test_size=split, stratify=y)
        if val_split != None:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, random_state=self.rs, test_size=val_split, stratify=self.y_train)
        else:
            self.X_val = None
            self.y_val = None

    def scale_data(self, columns):
        scaler = MinMaxScaler()
        self.data.loc[:,columns] = scaler.fit_transform(self.data.loc[:,columns])
        self.scaler = scaler

    def get_positive_indices(self, return_data=False):
        self.pos_indices = np.where(self.y_train == 1.0)[0]
        if return_data:
            return self.X_train[self.pos_indices]

    def get_negative_indices(self, return_data=False):
        self.neg_indices = np.where(self.y_train == 0.0)[0]
        if return_data:
            return self.X_train[self.neg_indices]

    def prepare_batches(self, batch_size, pos_probability = None, neg_probability = None):
        if len(self.X_train) % batch_size != 0:
            assert(f"Training set of length {len(self.X_train)} is not divisible by {batch_size}. Choose a different batch size if you would like to incoporate all data samples in training.")
        X_batches = []
        y_batches = []
        pi = self.pos_indices
        ni = self.neg_indices
        num_samples = len(self.X_train) // batch_size
        for _ in range(batch_size):
            selected__pos_indices = np.random.choice(pi, size=num_samples//2, replace=False, p=pos_probability)
            selected_neg_indices = np.random.choice(ni, size=num_samples//2, replace=False, p=neg_probability)
            selected_indices = np.sort(np.concatenate((selected__pos_indices, selected_neg_indices)))
            X_batches.append(self.X_train[selected_indices])
            y_batches.append(self.y_train[selected_indices])
        return X_batches, y_batches

class CounterfactuallyDebiasedModel(tf.keras.Model):
    def __init__(self, hidden_layers=[45, 36, 27, 18], output_size=1, num_epochs=50, batches=50, lr=0.0001):
        super(CounterfactuallyDebiasedModel, self).__init__()
        self.hl = hidden_layers
        self.num_epochs = num_epochs
        self.batches = batches
        self.lr = lr
        self.encoder_portion = self.encoder(hidden_layers, output_size)

    def encoder(self, hidden_layers, output_size, hidden_activation="relu", output_activation="sigmoid"):
        model = tf.keras.Sequential()
        for hl in hidden_layers:
            model.add(tf.keras.layers.Dense(hl, activation=hidden_activation))
        model.add(tf.keras.layers.Dense(output_size, activation=output_activation))
        return model

    def encode(self, x):
        return self.encoder_portion(x)
    
    def predict(self, x):
        yhat = self.encode(x)
        return tf.squeeze(yhat)

    @classmethod
    def custom_loss(cls, y, yhat):
        classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return classification_loss(y.reshape(-1, 1), yhat)

    def train(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            yhat = self.encode(x)
            loss = CounterfactuallyDebiasedModel.custom_loss(y.astype('float32'), yhat)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def fit(self, new_dataset):
        loss = 0
        optim = tf.keras.optimizers.Adam(self.lr)
        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            xs, ys = new_dataset.prepare_batches(self.batches)
            for batch_x, batch_y in zip(xs, ys): # apply np.random_choice and its non-uniform probabilities
                loss = self.train(batch_x, batch_y, optim)
        # print(f"Final loss: {loss}")

def eval(data, hyperparameters):
    model = CounterfactuallyDebiasedModel(**hyperparameters)
    model.fit(data)
    predictions = model.predict(data.X_val)
    predictions = np.array([0 if p < 0.5 else 1 for p in predictions])
    
    f1s = []
    for a in [11, 17, 18, 19, 20]:
        positive, negative = stratified_f1(data.X_val, data.y_val, predictions, a)
        combined = list(positive.values())
        combined.extend(list(negative.values()))
        f1s.append(np.mean(np.array(combined)))
    return np.mean(np.array(f1s))
    

def stratified_f1(x, truth, predictions, index):

    f1s_pos = {}
    f1s_neg = {}
    values = x[:,index]
    for i in np.unique(values):
        indices = np.where(x[:,index] == i)
        f1s_pos[i] = f1_score(truth[indices], predictions[indices], pos_label=1)

    for i in np.unique(values):
        indices = np.where(x[:,index] == i)
        f1s_neg[i] = f1_score(truth[indices], predictions[indices], pos_label=0)
    return f1s_pos, f1s_neg

if __name__ == "__main__":

    # commandline inputs
    args = parser.parse_args()

    # create dataset
    new_dataset = Dataset(args.input, args.predictor, args.scale, val_split=0.1)    
    # set hyperparameters
    threshold = 0.5

    hyperparameters = {"hidden_layers": [[45, 36, 27, 18]], "output_size": [args.output], "num_epochs": [50], "batches": [50], "lr": [0.0001]}
    bh, bscore = grid_search_multi(eval, new_dataset, 2, **hyperparameters)
    print(f"Best Score: {bscore}")
    print(f"Best hyperparameters: {bh}")
    print()

    # create model
    full_dataset = Dataset(args.input, args.predictor, args.scale)
    model = CounterfactuallyDebiasedModel(**bh)
    model.fit(full_dataset)
    
    # Evaluate model
    predictions = model.predict(full_dataset.X_test)
    predictions = np.array([0 if p < threshold else 1 for p in predictions])
    print(accuracy_score(full_dataset.y_test, predictions))
    print(f1_score(full_dataset.y_test, predictions))
    print(stratified_f1(full_dataset.X_test, full_dataset.y_test, predictions, 20))
    for name, df in zip(["./data_ca/X_train.csv", "./data_ca/X_test.csv", "./data_ca/y_train.csv", "./data_ca/y_test.csv", "./data_ca/y_predict.csv"], [full_dataset.X_train, full_dataset.X_test, full_dataset.y_train, full_dataset.y_test, pd.DataFrame(predictions)]):
        pd.DataFrame(df).to_csv(name)