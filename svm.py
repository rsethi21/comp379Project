from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input path to csv/data", required=True)
parser.add_argument("-o", "--output", help="output folder to metrics", required=False, default=".")
parser.add_argument("-p", "--hyperparameters", help="path to json file with hyperparameters to search against", required=False, default=None)
parser.add_argument("-t", "--target", help="name of the column that will serve as the target", required=True)
parser.add_argument('-s', '--scale', help='name of columns to scale', required=False, nargs="*", default=None)
parser.add_argument("-c1", "--testsplit", help="percent to split train and test data by", required=False, type=float, default=0.2)
parser.add_argument("-m", "--multiprocess", help="number of cores to use for multiprocessing", required=False, type=int, default=1)
parser.add_argument("-k", "--numcrossfolds", help="number of crossfolds for gridsearch if requested", required=False, type=int, default=5)
parser.add_argument("-e", "--evaluation", help="evaluation metric for gridsearch", required=False, default="f1")

class Dataset:
    def __init__(self, filepath, predictor, scale_columns, split=0.2, val_split=None, rs = 4):
        self.rs = rs
        self.open_dataset(filepath)
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

def create_model(hyperparameter=None):
    if hyperparameter != None:
        model = SVC(**hyperparameter)
    else:
        model = SVC()
    return model

def evaluate(y, yhat, folder_for_cm=None):

    f1_pos = f1_score(y, yhat, pos_label=1)
    f1_neg = f1_score(y, yhat, pos_label=0)
    acc = accuracy_score(y, yhat)
    if folder_for_cm != None:
        cm = confusion_matrix(y, yhat)
        figure = plt.figure()
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        figure.savefig(os.path.join(folder_for_cm, "confusion.png"))
        plt.close(figure)
    return f1_pos, f1_neg, acc

def search(data_obj, hyperparameter_sets, mp, cv, scoring):

    clf = create_model()
    search_results = GridSearchCV(clf, hyperparameter_sets, scoring=scoring, n_jobs=mp, cv=cv)
    search_results.fit(data_obj.X_train, data_obj.y_train)
    return search_results.best_estimator_, search_results.best_score_

def save(model_obj, data_obj, folderpath):
    
    with open(os.path.join(folderpath, "model.pkl"), "wb") as model_file:
        pickle.dump(model_obj, model_file)

    with open(os.path.join(folderpath, "dataset.pkl"), "wb") as dataset_file:
        pickle.dump(data_obj, dataset_file)

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

    args = parser.parse_args()
    dataset = Dataset(args.input, args.target, args.scale, split=args.testsplit)
    
    if args.hyperparameters != None:
        with open(args.hyperparameters, "r") as hyperparameter_file:
            search_hyperparameters = json.load(hyperparameter_file)
        model, score = search(dataset, search_hyperparameters, args.multiprocess, args.numcrossfolds, args.evaluation)
    else:
        model = create_model()
        model.fit(dataset.X_train, dataset.y_train)
    
    print(evaluate(dataset.y_test, model.predict(dataset.X_test), args.output))
    print(stratified_f1(dataset.X_test, dataset.y_test, model.predict(dataset.X_test), 20))
    save(model, dataset, args.output)