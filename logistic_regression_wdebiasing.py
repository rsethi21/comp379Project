import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("balanced_data.csv")

#specify the columns to scale
columns_to_scale = ['BMI', 'Age', 'Education', 'GenHlth', 'MentHlth', 'PhysHlth']

#extract the target variable and the feature variables
y = df['Diabetes_binary']
X = df[['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education', 'Income']]


# extract the features to scale
X_to_scale = X[columns_to_scale]
#Extract the featurs that don't require scaling
X_not_to_scale = X.drop(columns=columns_to_scale)


scaler = MinMaxScaler()
#scale the desired columns
X_to_scale = pd.DataFrame(scaler.fit_transform(X_to_scale), columns=columns_to_scale)

#concatenate the scaled and unscaled features
X_scaled = pd.concat([X_not_to_scale, X_to_scale], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



C_values = [0.001, 0.01, 0.1, 1, 10]
penalty_values = ['l1', 'l2']


k = 5

num_samples = len(y_train)

indices = np.arange(num_samples)
np.random.shuffle(indices)

fold_size = num_samples // k

best_f1_score = 0.0
best_params = {'C': None, 'penalty': None}
all_f1_scores = []

for C in C_values:

    for penalty in penalty_values:
        # Perform k-fold cross-validation
        f1_scores = []
        for i in range(k):
         # Define the start and end indices for the test fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < k - 1 else num_samples

            # Use the remaining data as training folds
            train_indices = np.concatenate((indices[:test_start], indices[test_end:]))

            # Use the current fold as the test fold
            test_indices = indices[test_start:test_end]

            # Assuming model is your classifier and is already initialized
            model = LogisticRegression(C = C, penalty = penalty, solver = 'saga')

            # Train your model on k-1 folds
            model.fit(X_train.iloc[train_indices], y_train.iloc[train_indices])

            # Test your model on the remaining fold
            y_pred_fold = model.predict(X_train.iloc[test_indices])

            f1_fold = f1_score(y_train.iloc[test_indices], y_pred_fold)

            # Append the F1 score to the list
            f1_scores.append(f1_fold)

        mean_f1 = np.mean(f1_scores)

        if mean_f1 > best_f1_score:
            best_f1_score = mean_f1
            best_params['C'] = C
            best_params['penalty'] = penalty

        all_f1_scores.append({'C': C, 'penalty': penalty, 'mean_f1': mean_f1})

print("Best Hyperparameters:", best_params)
print("Best training F1 Score:", best_f1_score)



logistic = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver = 'saga')
logistic.fit(X_test, y_test)

predictions = logistic.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)



print("Test Accuracy:", accuracy)
print("Test F1 Score:", f1)

df_eval = pd.DataFrame({'Income': X_test['Income'], 'True_Label': y_test, 'Predicted_Label': predictions})

# Create a list of unique income groups
income_groups = sorted(df_eval['Income'].unique())

# Initialize lists to store FP and FN values for each income group
FP_values = []
FN_values = []


# Iterate over income groups and calculate FP and FN for each group
for income_group in income_groups:
    group_data = df_eval[df_eval['Income'] == income_group]
    group_true_labels = group_data['True_Label']
    group_predicted_labels = group_data['Predicted_Label']
    confusion = confusion_matrix(group_true_labels, group_predicted_labels)
    TN, FP, FN, TP = confusion.ravel()
    
    # Append FP and FN values to the lists
    FP_values.append(FP)
    FN_values.append(FN)

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(income_groups))

plt.bar(index, FP_values, bar_width, label='False Positives')
plt.bar([i + bar_width for i in index], FN_values, bar_width, label='False Negatives')

plt.xlabel('Income Group')
plt.ylabel('Count')
plt.title('False Positives and False Negatives by Income Group')
plt.xticks([i + bar_width / 2 for i in index], income_groups)
plt.legend()

plt.show()

df_eval['True_Label'] = df_eval['True_Label'].astype(int)
df_eval['Predicted_Label'] = df_eval['Predicted_Label'].astype(int)

# Calculate the F1 score for each income bracket
income_f1_scores = df_eval.groupby('Income').apply(lambda x: f1_score(x['True_Label'], x['Predicted_Label'])).rename('F1_Score')

plt.figure(figsize=(10, 6))
plt.bar(income_groups, income_f1_scores, color='blue')
plt.xlabel('Income Level')
plt.ylabel('F1 Score')
plt.title('F1 Scores by Income Level')
plt.xticks(rotation=45)
plt.show()

total_individuals_by_income = X_test.groupby('Income').size()
misclassifications = (y_test != predictions)

misclass_df = pd.DataFrame({'Income': X_test['Income'], 'Misclassification': misclassifications})
misclass_counts = misclass_df.groupby(['Income'])['Misclassification'].sum()

misclassification_rates = misclass_counts / total_individuals_by_income

print("Misclassifications by Income Level:")
print(misclass_counts)
print("Misclassification Rates by Income Level:")
print(misclassification_rates)
print(total_individuals_by_income)

df_eval = pd.DataFrame({'Income': X_test['Income'], 'True_Label': y_test, 'Predicted_Label': predictions})

# Create a list of unique income groups
income_groups = sorted(df_eval['Income'].unique())

# Iterate over income groups and calculate FP and FN for each group
for income_group in income_groups:
    group_data = df_eval[df_eval['Income'] == income_group]
    group_true_labels = group_data['True_Label']
    group_predicted_labels = group_data['Predicted_Label']
    confusion = confusion_matrix(group_true_labels, group_predicted_labels)
    TN, FP, FN, TP = confusion.ravel()
    
    # Print or store FP and FN values for each income group
    print(f"Income Group: {income_group}")
    print("False Positives (FP):", FP)
    print("False Negatives (FN):", FN)

feature_names = X_train.columns

coefficients = logistic.coef_[0]

feature_weights_df = pd.DataFrame({'Feature': feature_names, 'Weight': coefficients})

df_eval['True_Label'] = df_eval['True_Label'].astype(int)
df_eval['Predicted_Label'] = df_eval['Predicted_Label'].astype(int)

# Calculate the F1 score for each income bracket
income_f1_scores = df_eval.groupby('Income').apply(lambda x: f1_score(x['True_Label'], x['Predicted_Label'])).rename('F1_Score')

# Print the F1 scores
print("F1 score for each income bracket:")
print(income_f1_scores)

print("Features and their Weights:")
print(feature_weights_df)

#plot the number of samples from each income group (8 = highest income)
plt.figure(figsize=(10, 6))
sns.countplot(x='Income', data=df)
plt.title('Distribution of People in Income Groups')
plt.xlabel('Income Group')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))

# 'hue' parameter is used to differentiate between positive and negative classes
sns.countplot(x='Income', hue='Diabetes_binary', data=df)

plt.title('Distribution of Positives and Negatives in Income Groups')
plt.xlabel('Income Group')
plt.ylabel('Count')
plt.legend(title='Class', labels=['Negative', 'Positive'])  # Update labels as per your data

plt.show()
