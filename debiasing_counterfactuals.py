# This model file is adapted from: © MIT 6.S191: Introduction to Deep Learning http://introtodeeplearning.com

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


# print(tf.config.list_physical_devices("GPU"))


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


