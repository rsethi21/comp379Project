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

# print(tf.config.list_physical_devices("GPU"))


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input path to diabetes file", required=True)
parser.add_argument("-p", "--predictor", help="name of prediction column", required=True)
parser.add_argument("-s", "--scale", help="columns to scale", nargs="*", required=False, default=[])
parser.add_argument("-n", "--nodes", help="list of hidden layer nodes", nargs="*", required=True, type=int)
parser.add_argument("-o", "--ouput", help="number of output nodes", required=False, type=int, default=0)

class Dataset:
    def __init__(self, filepath, predictor, scale_columns, split=0.2, rs = 2023):
        self.rs = rs
        self.open_dataset(filepath)
        self.scale_data(scale_columns)
        self.separate_samples(self.data, predictor, split)

    def open_dataset(self, filepath):
        self.data = pd.read_csv(filepath, header = 0)

    def separate_samples(self, data, predictor, split):
        self.columns = data.columns
        data = data.dropna()
        data = data.sample(frac=1, random_state=self.rs)
        y = data.loc[:,predictor].to_numpy()
        X = data.drop(columns=[predictor]).to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=self.rs, test_size=split)

    def scale_data(self, columns):
        scaler = MinMaxScaler()
        self.data.loc[:,columns] = scaler.fit_transform(self.data.loc[:,columns])
        self.scaler = scaler

    def prepare_batches(self, batch_size):
        X_batches, y_batches = np.array_split(self.X_train, batch_size), np.array_split(self.y_train, batch_size)
        return X_batches, y_batches

class FFM(tf.keras.Model):
    def __init__(self, hidden_layers, output_size):
        super(FFM, self).__init__()
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
        return yhat

    @classmethod
    def custom_loss(cls, y, yhat):
        classification_loss = tf.nn.CategoricalCrossentropy(y.reshape(-1, 1), yhat)
        return classification_loss

    def train(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            yhat = self.encode(x)
            loss = FFM.custom_loss(y.astype('float32'), yhat)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

def stratified_f1(x, truth, predictions, index):

    f1s = {}
    values = x[:,index]
    for i in np.unique(values):
        indices = np.where(x[:,index] == i)
        f1s[i] = f1_score(truth[indices], predictions[indices])
    return f1s

if __name__ == "__main__":

    # commandline inputs
    args = parser.parse_args()

    # create dataset
    new_dataset = Dataset(args.input, args.predictor, args.scale)
    
    # set hyperparameters
    batches = 100
    lr = 0.0001
    num_epochs = 100
    optim = tf.keras.optimizers.Adam(lr)
    threshold = 0.5

    # create model
    model = FFM(args.nodes, args.output)
    
    # training loop
    loss = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")
        xs, ys = new_dataset.prepare_batches(batches)
        for batch_x, batch_y in tqdm(zip(xs, ys)): # apply np.random_choice and its non-uniform probabilities
            loss = model.train(batch_x, batch_y, optim)
        print(f"Latest loss: {loss}")
    
    # Evaluate model
    predictions = tf.squeeze(model.predict(new_dataset.X_test))
    predictions = np.array([0 if p < threshold else 1 for p in predictions])
    print(accuracy_score(new_dataset.y_test, predictions))
    print(f1_score(new_dataset.y_test, predictions))
    print(stratified_f1(new_dataset.X_test, new_dataset.y_test, predictions, 20))
    model.save_weights("model_weights_2/ff")