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

# print(tf.config.list_physical_devices("GPU"))


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input path to diabetes file", required=True)
parser.add_argument("-p", "--predictor", help="name of prediction column", required=True)
parser.add_argument("-s", "--scale", help="columns to scale", nargs="*", required=False, default=[])

class Dataset:
    def __init__(self, filepath, predictor, scale_columns, split=0.2, rs = 4):
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=self.rs, test_size=split, stratify=y)

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
    def __init__(self, latent_dim, hidden_layers, input_size):
        super(DebiasedModel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_output = 2*self.latent_dim + 1
        self.encoder_portion = self.encoder(hidden_layers)
        self.decoder_portion = self.decoder(hidden_layers, input_size)
        
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
        means = encoded_output[:,1:self.latent_dim+1]
        sigmas = encoded_output[:,self.latent_dim+1:]
        return yhat, means, sigmas

    def decode(self, latent_vector, input_size):
        return self.decoder_portion(latent_vector, input_size)
    
    def run(self, x):
        yhat, means, sigmas = self.encode(x)
        latent_vector = self.extract_latent_vector(means, sigmas)
        xhat = self.decode(latent_vector, len(x[0]))
        return yhat, means, sigmas, xhat
    
    def predict(self, x):
        yhat, means, sigmas = self.encode(x)
        return tf.squeeze(tf.sigmoid(yhat))

    def extract_latent_vector(self, means, sigmas):
        return DebiasedModel.sampling(means, sigmas)

    @classmethod
    def sampling(cls, mean, sigma):
        epsilon = tf.random.normal(shape=mean.shape)
        return mean + tf.math.exp(0.5 * sigma) * epsilon

    @classmethod
    def custom_loss(cls, x, xhat, y, yhat, mu, sigma, debias_weight):
        latent_normalization_loss = 0.5 * tf.reduce_sum(tf.exp(sigma) + tf.square(mu) - 1.0 - sigma, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.abs(x-xhat), axis=1)
        classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(y.reshape(-1, 1), yhat)
        vae_loss = reconstruction_loss + debias_weight * latent_normalization_loss
        total_loss = tf.reduce_mean(classification_loss + vae_loss)
        return total_loss, classification_loss

    def extract_distributions(self, data, latent_dim, num_batches):
        mu = np.zeros((len(data), latent_dim))
        for start_i in range(0, len(data), len(data)//num_batches):
            end_i = min(start_i+len(data)//num_batches, len(data)+1)
            _, batch_mu, _ = self.encode(data[start_i:end_i])
            mu[start_i:end_i] = batch_mu

        training_sample_p = np.zeros(mu.shape[0])
        for i2 in range(latent_dim):
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

    def stratied_train(self, x, y, optimizer, weight):
        with tf.GradientTape() as tape:
            yhat_p, mean_p, sigma_p, xhat_p = self.run(x[np.where(y == 1)[0]])
            yhat_n, mean_n, sigma_n, xhat_n = self.run(x[np.where(y == 0)[0]])
            loss_p, class_loss_p = DebiasedModel.custom_loss(x[np.where(y == 1)[0]], xhat_p, y[np.where(y==1)[0]].astype('float32'), yhat_p, mean_p, sigma_p, weight)
            loss_n, class_loss_n = DebiasedModel.custom_loss(x[np.where(y == 0)[0]], xhat_n, y[np.where(y==0)[0]].astype('float32'), yhat_n, mean_n, sigma_n, weight)
            loss = tf.reduce_mean(loss_p + loss_n)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def train(self, x, y, optimizer, weight):
        with tf.GradientTape() as tape:
            yhat, mean, sigma, xhat = self.run(x)
            loss, class_loss = DebiasedModel.custom_loss(x, xhat, y.astype('float32'), yhat, mean, sigma, weight)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
    def fit(self, num_epochs, new_dataset, batches, latent_features, optim, db_weight):
        loss = 0
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            pos_probs = self.extract_distributions(new_dataset.get_positive_indices(return_data=True), latent_features, batches)
            neg_probs = self.extract_distributions(new_dataset.get_negative_indices(return_data=True), latent_features, batches)
            xs, ys = new_dataset.prepare_batches(batches, pos_probability=pos_probs, neg_probability=neg_probs)
            for batch_x, batch_y in zip(xs, ys): # apply np.random_choice and its non-uniform probabilities
                loss = self.train(batch_x, batch_y, optim, db_weight)
        print(f"Final loss: {loss}")


def search(model, data, hl, **hyperparameters):
    pass

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
    new_dataset = Dataset(args.input, args.predictor, args.scale)
    
    # set hyperparameters
    batches = 100
    lr = 0.0001
    latent_features = 4 # hx, sx, ses, demographics
    num_epochs = 100
    optim = tf.keras.optimizers.Adam(lr)
    threshold = 0.5
    db_weight = 0.0005
    hl = [18, 15, 12, 9]

    # create model
    model = DebiasedModel(latent_features, hl, len(new_dataset.columns)-1)
    model.fit(num_epochs, new_dataset, batches, latent_features, optim, db_weight)
    # current approach takes the normalized overall latent space and extracts distribution of latent space for each indicator from that overall distribution
    # possible extension is to map the latent space distribution for each one normally and then pick probabilities
    
    # Evaluate model
    predictions = model.predict(new_dataset.X_test)
    predictions = np.array([0 if p < threshold else 1 for p in predictions])
    print(accuracy_score(new_dataset.y_test, predictions))
    print(f1_score(new_dataset.y_test, predictions))
    print(stratified_f1(new_dataset.X_test, new_dataset.y_test, predictions, 20))
    model.save_weights("model_weights/dbvae")
    for name, df in zip(["./data_db/X_train.csv", "./data_db/X_test.csv", "./data_db/y_train.csv", "./data_db/y_test.csv", "./data_db/y_predict.csv"], [new_dataset.X_train, new_dataset.X_test, new_dataset.y_train, new_dataset.y_test, pd.DataFrame(predictions)]):
        pd.DataFrame(df).to_csv(name)