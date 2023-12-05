# This model file is adapted and modified from: © MIT 6.S191: Introduction to Deep Learning http://introtodeeplearning.com

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


class DebiasedModel(tf.keras.Model):
    def __init__(self, latent_dim=6, hidden_layers=[18, 15, 12, 9], input_size=21, db_weight0=0.0005, db_weight1=0.0005, num_epochs=50, batches=50, lr=0.0001):
        super(DebiasedModel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_output = 4*self.latent_dim + 1
        self.encoder_portion = self.encoder(hidden_layers)
        self.decoder_portion = self.decoder(hidden_layers, input_size)
        self.db_weight0 = db_weight0
        self.db_weight1 = db_weight1
        self.num_epochs = num_epochs
        self.batches = batches
        self.lr = lr
        
    def decoder(self, hidden_layers, input_size, hidden_activation="relu"):
        model = tf.keras.Sequential()
        for rhl in reversed(hidden_layers):
            model.add(tf.keras.layers.Dense(rhl, activation=hidden_activation))
        model.add(tf.keras.layers.Dense(input_size, activation=None))
        return model

    def encoder(self, hidden_layers, hidden_activation="relu"):
        model = tf.keras.Sequential()
        for hl in hidden_layers:
            model.add(tf.keras.layers.Dense(hl, activation=hidden_activation))
        model.add(tf.keras.layers.Dense(self.encoder_output, activation=None))
        return model

    def encode(self, x):
        encoded_output = self.encoder_portion(x)
        yhat = tf.expand_dims(encoded_output[:,0],-1)
        mean0 = encoded_output[:,1:self.latent_dim+1]
        mean1 = encoded_output[:,self.latent_dim+1:2*self.latent_dim+1]
        sigma0 = encoded_output[:,2*self.latent_dim+1:3*self.latent_dim+1]
        sigma1 = encoded_output[:,3*self.latent_dim+1:]
        return yhat, mean0, mean1, sigma0, sigma1

    def decode(self, latent_vector, input_size):
        return self.decoder_portion(latent_vector, input_size)
    
    def run(self, x):
        yhat, mean0, mean1, sigma0, sigma1 = self.encode(x)
        latent_vector0 = self.extract_latent_vector(mean0, sigma0)
        latent_vector1 = self.extract_latent_vector(mean1, sigma1)
        xhat0 = self.decode(latent_vector0, len(x[0]))
        xhat1 = self.decode(latent_vector1, len(x[0]))
        return yhat, mean0, mean1, sigma0, sigma1, xhat0, xhat1
    
    def predict(self, x):
        yhat, mean0, mean1, sigma0, sigma1 = self.encode(x)
        return tf.squeeze(tf.sigmoid(yhat))

    def extract_latent_vector(self, means, sigmas):
        return DebiasedModel.sampling(means, sigmas)

    @classmethod
    def sampling(cls, mean, sigma):
        epsilon = tf.random.normal(shape=mean.shape)
        return mean + tf.math.exp(0.5 * sigma) * epsilon

    @classmethod
    def custom_loss(cls, x, xhat0, xhat1, y, yhat, mu0, mu1, sigma0, sigma1, debias_weight0, debias_weight1):
        latent_normalization_loss0 = 0.5 * tf.reduce_sum(tf.exp(sigma0) + tf.square(mu0) - 1.0 - sigma0, axis=1)
        reconstruction_loss0 = tf.reduce_mean(tf.abs(x-xhat0), axis=1)

        latent_normalization_loss1 = 0.5 * tf.reduce_sum(tf.exp(sigma1) + tf.square(mu1) - 1.0 - sigma1, axis=1)
        reconstruction_loss1 = tf.reduce_mean(tf.abs(x-xhat1), axis=1)

        classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(y.reshape(-1, 1), yhat)
        
        vae_loss0 = reconstruction_loss0 + debias_weight0 * latent_normalization_loss0
        vae_loss1 = reconstruction_loss1 + debias_weight1 * latent_normalization_loss1
        indicator = tf.cast(tf.equal(y, 1), tf.float32)

        total_loss = tf.reduce_mean(classification_loss + (1-indicator)*vae_loss0 + indicator*vae_loss1)
        return total_loss, classification_loss

    def extract_distributions(self, data, num_batches, pos=False):
        mu = np.zeros((len(data), self.latent_dim))
        for start_i in range(0, len(data), len(data)//num_batches):
            end_i = min(start_i+len(data)//num_batches, len(data)+1)
            _, mean0, mean1, _, _ = self.encode(data[start_i:end_i])
            if pos:
                mu[start_i:end_i] = mean1
            else:
                mu[start_i:end_i] = mean0

        training_sample_p = np.zeros(mu.shape[0])
        for i2 in range(self.latent_dim):
            # revisit this loop
            latent_distribution = mu[:,i2]
            histo, bins = np.histogram(latent_distribution, density=True, bins=10)
            bins[0] = -float("inf")
            bins[-1] = float("inf")
            bins_idx = np.digitize(latent_distribution, bins)
            
            
            hist_smoothed_density = histo + 0.001
            hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)
            p = 1.0/(hist_smoothed_density[bins_idx-1])
            p = p / np.sum(p)
            training_sample_p = np.maximum(p, training_sample_p)

        training_sample_p /= np.sum(training_sample_p)

        return training_sample_p

    def stratified_train(self, x, y, optimizer, weight0, weight1):
        with tf.GradientTape() as tape:
            yhat, mean_p, mean_n, sigma_p, sigma_n, xhat_p, xhat_n = self.run(x)
            loss, class_loss_p = DebiasedModel.custom_loss(x, xhat_p, xhat_n, y.astype('float32'), yhat, mean_p, mean_n, sigma_p, sigma_n, weight0, weight1)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def fit(self, new_dataset):
        loss = 0
        optim = tf.keras.optimizers.Adam(self.lr)
        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            pos_probs = self.extract_distributions(new_dataset.get_positive_indices(return_data=True), self.batches, pos=True)
            neg_probs = self.extract_distributions(new_dataset.get_negative_indices(return_data=True), self.batches, pos=False)
            xs, ys = new_dataset.prepare_batches(self.batches, pos_probability=pos_probs, neg_probability=neg_probs)
            for batch_x, batch_y in zip(xs, ys): # apply np.random_choice and its non-uniform probabilities
                loss = self.stratified_train(batch_x, batch_y, optim, self.db_weight0, self.db_weight1)
        print(f"Final loss: {loss}")

def eval(data, hyperparameters):
    model = DebiasedModel(**hyperparameters)
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
    
    threshold = 0.5
    hyperparameters = {"latent_dim": [2, 4, 6], "hidden_layers": [[45, 36, 27, 18]], "input_size": [21], "db_weight0":[0.1, 0.001, 0.0001, 0.00001], "db_weight1":[0.1, 0.001, 0.0001, 0.00001], "batches":[50], "num_epochs": [50], "lr":[0.0001]}


    # simple hyperparameter search
    bh, bscore = grid_search_multi(eval, new_dataset, 4, **hyperparameters)
    print(f"Best Score: {bscore}")
    print(f"Best hyperparameters: {bh}")
    print()

    # create model
    full_dataset = Dataset(args.input, args.predictor, args.scale)
    model = DebiasedModel(**bh)
    model.fit(full_dataset)
    
    # Evaluate model
    predictions = model.predict(full_dataset.X_test)
    predictions = np.array([0 if p < threshold else 1 for p in predictions])
    print(accuracy_score(full_dataset.y_test, predictions))
    print(f1_score(full_dataset.y_test, predictions))
    print(stratified_f1(full_dataset.X_test, full_dataset.y_test, predictions, 20))
    for name, df in zip(["./data_db/X_train.csv", "./data_db/X_test.csv", "./data_db/y_train.csv", "./data_db/y_test.csv", "./data_db/y_predict.csv"], [full_dataset.X_train, full_dataset.X_test, full_dataset.y_train, full_dataset.y_test, pd.DataFrame(predictions)]):
        pd.DataFrame(df).to_csv(name)