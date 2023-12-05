import argparse
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.utils import shuffle
import pdb
from gridsearch import grid_search_multi
import os

# print(tf.config.list_physical_devices("GPU"))


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input path to diabetes file", required=True)
parser.add_argument("-p", "--predictor", help="name of prediction column", required=True)
parser.add_argument("-s", "--scale", help="columns to scale", nargs="*", required=False, default=[])
parser.add_argument("-o", "--output", help="number of output nodes", required=False, type=int, default=1)
parser.add_argument("-v", "--hyperparameters", help="path to json file with hyperparameters to search against", required=False, default=None)
parser.add_argument("-k", "--kfolds", help="number of folds for cross validation", required=False, type=int, default=5)
parser.add_argument("-m", "--multiprocess", help="number of processors to use for selection", required=False, type=int, default=1)
parser.add_argument("-c", "--percent", help="percent split of test and train", required=False, type=float, default=0.2)
parser.add_argument("-d", "--directory", help="path to folder to save model session", required=False, default=".")

class Dataset:
    def __init__(self, filepath, predictor, scale_columns, split=0.2, val_split=None, rs = 4):
        self.rs = rs
        self.open_dataset(filepath)
        if len(scale_columns) != 0:
            self.scale_data(scale_columns)
        self.separate_samples(self.data, predictor, split)

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

    def prepare_batches(self, batch_size):
        X_batches, y_batches = np.array_split(self.X_train, batch_size), np.array_split(self.y_train, batch_size)
        return X_batches, y_batches

class FFM(tf.keras.Model):
    def __init__(self, hidden_layers=[18, 15, 12, 9], output_size=1, num_epochs=50, batches=50, lr=0.0001):
        super(FFM, self).__init__()
        self.encoder_portion = self.encoder(hidden_layers, output_size)
        self.num_epochs = num_epochs
        self.batches = batches
        self.lr = lr

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
            loss = FFM.custom_loss(y.astype('float32'), yhat)
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


def search(new_dataset, cv, hyperparameters, numprocessors):
    scores = []
    hp = []
    original_X_train = np.copy(new_dataset.X_train)
    original_y_train = np.copy(new_dataset.y_train)
    splits = KFold(n_splits=cv, shuffle=False)
    for i, (train_index, test_index) in enumerate(splits.split(original_X_train)):
        print(f"Fold: {i+1}...")
        new_dataset.X_train = original_X_train[train_index]
        new_dataset.y_train = original_y_train[train_index]
        new_dataset.X_val = original_X_train[test_index]
        new_dataset.y_val = original_y_train[test_index]
        _, _, scorings, hs = grid_search_multi(eval, new_dataset, numprocessors, return_all=True, **hyperparameters)
        scores.append(list(scorings))
        hp = hs
    new_dataset.X_train = original_X_train
    new_dataset.y_train = original_y_train
    new_dataset.X_val = None
    new_dataset.y_val = None
    best_score = np.mean(np.array(scores), axis=0)
    i = np.argmax(best_score)
    return best_score[i], hp[i]

def eval(data, hyperparameters):
    model = FFM(**hyperparameters)
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
    new_dataset = Dataset(args.input, args.predictor, args.scale, args.percent)    
    # set hyperparameters
    threshold = 0.5

    if args.hyperparameters != None:
        with open(args.hyperparameters, "r") as json_file:
            hyperparameters = json.load(json_file)

    # bscore, bh = search(new_dataset, args.kfolds, hyperparameters, args.multiprocess, args.percent)
    # print(f"Best Score: {bscore}")
    # print(f"Best hyperparameters: {bh}")
    # print()

    bh = {"hidden_layers": [45, 36, 27, 18], "output_size": args.output, "num_epochs": 50, "batches": 50, "lr": 0.0001}

    # create model
    model = FFM(**bh)
    model.fit(new_dataset)
    
    # Evaluate model
    predictions = model.predict(new_dataset.X_test)
    predictions = np.array([0 if p < threshold else 1 for p in predictions])
    print(accuracy_score(new_dataset.y_test, predictions))
    print(f1_score(new_dataset.y_test, predictions))
    print(stratified_f1(new_dataset.X_test, new_dataset.y_test, predictions, 20))
    for name, df in zip([os.path.join(args.directory, "X_train.csv"), os.path.join(args.directory, "X_test.csv"), os.path.join(args.directory, "y_train.csv"), os.path.join(args.directory, "y_test.csv"), os.path.join(args.directory, "y_predict.csv")], [new_dataset.X_train, new_dataset.X_test, new_dataset.y_train, new_dataset.y_test, pd.DataFrame(predictions)]):
        pd.DataFrame(df).to_csv(name)