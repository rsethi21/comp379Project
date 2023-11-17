import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input file path", required=True)
parser.add_argument("-p", "--processors", help="number of processors to use for search", required=True)
parser.add_argument("-o", "--output", help="output folder path", required=False, default=".")


if __name__ == "__main__":
    
    args = parser.parse_args()
    random_state = 4

    data = pd.read_csv(args.input, header=0)
    data = data.sample(frac = 1, random_state=random_state)

    y = data.Diabetes_binary
    X = data.drop(columns=["Diabetes_binary"])

    scaler = MinMaxScaler()
    X.loc[:, ["BMI", "MentHlth", "PhysHlth"]] = scaler.fit_transform(X.loc[:, ["BMI", "MentHlth", "PhysHlth"]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.2)

    X_train.to_csv(os.path.join(args.output, "X_train.csv"))
    X_test.to_csv(os.path.join(args.output, "X_test.csv"))
    y_train.to_csv(os.path.join(args.output, "y_train.csv"))
    y_test.to_csv(os.path.join(args.output, "y_test.csv"))

    svm = SVC(random_state=random_state)
    parameters = {'kernel': ["sigmoid", "rbf", "linear", "poly"], "C": [10**-3, 10**-1, 1, 10**1, 10**3]}

    best_model = GridSearchCV(estimator=svm, param_grid=parameters, scoring='f1', verbose=4, n_jobs=int(args.processors), cv=5)
    best_model.fit(X_train, y_train)

    print(f"Best f1-score: {best_score_}")
    print(f"Best model params: {best_model.best_params_}")
    retrained_best_model = SVC(random_state=random_state, **best_model.best_params_)
    retrained_best_model.fit(X_train, y_train)

    cm = confusion_matrix(y_test.to_numpy().flatten(), retrained_best_model(X_test))
    score = accuracy_score(y_test.to_numpy().flatten(), retrained_best_model(X_test))
    print(score)

    with open(os.path.join(args.output, "best_svm_model.pkl"), "wb") as model_file:
        pickle.dump(retrained_best_model, model_file)

    fig = ConfusionMatrixDisplay(cm).figure_
    fig.savefig(os.path.join(args.output, "confusion_matrix.png"))
