# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


# Load breast cancer dataset
#/Users/manyasingh/Desktop/COMP479/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
file_path = '/Users/manyasingh/Desktop/COMP479/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
df = pd.read_csv(file_path)
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

scaler = MinMaxScaler()
X.loc[:, ["BMI", "MentHlth", "PhysHlth"]] = scaler.fit_transform(X.loc[:, ["BMI", "MentHlth", "PhysHlth"]])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features by scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Choose the value of k for KNN (you can experiment with different values)
k = 3

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train.values.ravel())

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

