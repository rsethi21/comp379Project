import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_excel("/Users/haarisanjum/Downloads/Diabetes.xlsx")

df['Income'] = df['Income'].astype(int)

# Select features to scale
features_to_scale = ['BMI', 'PhysHlth', 'MentHlth']

# Create a copy of the dataframe to avoid modifying the original
df_scaled = df.copy()

# Feature selection
selected_features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                     'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
                     'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Income']

X = df_scaled[selected_features]
y = df_scaled['Diabetes_binary']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale selected features using MinMaxScaler
scaler = MinMaxScaler()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Define the perceptron model
perceptron = Perceptron()

# Define the hyperparameter grid for grid search
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [10000, 25000, 50000]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(perceptron, param_grid, verbose = 5, scoring='f1', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Extracts the best hyperparameters and corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predictions are made using the model on the testing set
y_pred = best_model.predict(X_test)

# Metrics of evaluation like accuracy and f1 score are found (confusion matrix also included)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Prints the results
print("Best Hyperparameters:", best_params)
print("Best F1 Score:", f1)
print("Best model testing data")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Misclassifications by Income Level:")
misclassifications_by_income = pd.DataFrame(conf_matrix, columns=['0', '1'], index=['0', '1'])
print(misclassifications_by_income)

# Plot false positives and false negatives by income group
income_groups = df['Income'].unique()

false_positives = []
false_negatives = []

for group in income_groups:
    # Count false positives (misclassifications of the positive class)
    false_positives.append(np.sum((y_pred == 1) & (y_test == 0) & (X_test['Income'] == group)))

    # Count false negatives (misclassifications of the negative class)
    false_negatives.append(np.sum((y_pred == 0) & (y_test == 1) & (X_test['Income'] == group)))

plt.figure(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(income_groups))

plt.bar(index, false_positives, bar_width, color='blue', label='False Positives')
plt.bar(index + bar_width, false_negatives, bar_width, color='orange', label='False Negatives')

plt.xlabel('Income Group')
plt.ylabel('Count')
plt.title('False Positives and False Negatives by Income Group (w/ NoDocbcCost Dropped)')

plt.legend()

plt.show()

f1_scores = []

for group in income_groups:
    # Filter the predictions and labels for the current income group
    y_group = y_test[X_test['Income'] == group]
    y_pred_group = y_pred[X_test['Income'] == group]

    # Calculate F1 score for the current income group
    f1 = f1_score(y_group, y_pred_group)

    # Append the F1 score to the list
    f1_scores.append(f1)

# Plotting the bar graph
plt.bar(income_groups, f1_scores, color=['blue' for _ in range(len(income_groups))])
plt.xlabel('Income Group')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each Income Group')
plt.show()