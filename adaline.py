#!/usr/bin/env python
# coding: utf-8

# In[39]:


# importing sklearn packages, pandas, and numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# reading data
data = pd.read_excel('diabetes_binary_5050split_health_indicators_BRFSS2015.xlsx')


# checking for class balance
print(data['Diabetes_binary'].value_counts())
plt.bar(sorted(data['Diabetes_binary'].unique()), data['Diabetes_binary'].value_counts())
plt.xticks([])
x_labs = ['No', 'Yes']
plt.xticks([0, 1], x_labs)
plt.xlabel('Diabetes')
plt.ylabel('Count')
plt.title('Class Balance of Target Variable (Diabetes)')
plt.legend()
plt.show()

# converting 0 to -1 
data['Diabetes_binary'] = np.where(data['Diabetes_binary'] == 1, 1, -1)


# scaling BMI, MentHlth, PhysHlth
scaler = MinMaxScaler()
data[['BMI', 'MentHlth', 'PhysHlth']] = scaler.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])

# randomly splitting into 80% train, 20% test
train = data.sample(frac = 0.8, random_state = 0)
test = data.drop(train.index)

# defining a function to split train data into n subfolds
def get_n_folds(n, df):
    # randomly shuffle using df.sample
    df = df.sample(frac = 1, random_state = 0)
    # actually split df
    sub_dataframe = np.array_split(df, n)
    return sub_dataframe


# using the textbook implementation (Sebastian Raschka)
# NOT MY IMPLEMENTATION--I DO NOT TAKE CREDIT FOR THIS
class AdalineGD(object):
    def __init__(self, eta=0.000001, n_iter=10000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.activation(X)
            
            # Cost function
            error = (y - output)
            cost = (error**2).sum() / 2.0
            self.cost_.append(cost)
            
            # Update rule
            # print(self.w_[0], self.eta, error.sum())
            self.w_[1:] += self.eta * X.T.dot(error)
            self.w_[0] += self.eta * error.sum()
            
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

### training the intial model without any debiasing
adaline = AdalineGD(eta = 0.0000001, n_iter = 100000, random_state = 1)
adaline.fit(train.drop('Diabetes_binary', axis = 1), train['Diabetes_binary'])

# getting F1 and accuracy scores on the training data
y_pred_train = adaline.predict(train.drop('Diabetes_binary', axis = 1))
train_f1 = f1_score(train['Diabetes_binary'], y_pred_train)
train_accuracy = accuracy_score(train['Diabetes_binary'], y_pred_train)
print(f'Training F1 Score: {train_f1}')
print(f'Training Accuracy Score: {train_accuracy}')

# gettig F1 and accuracy scores on the test data
y_pred_test = adaline.predict(test.drop('Diabetes_binary', axis = 1))
test_f1 = f1_score(test['Diabetes_binary'], y_pred_test)
test_accuracy = accuracy_score(test['Diabetes_binary'], y_pred_test)
print(f'Testing F1 Score: {test_f1}')
print(f'Testing Accuracy Score: {test_accuracy}')


### graphing F1 and accuracy scores (prior to debiasing)

# Calculate the F1 score for each income bracket
df_eval = pd.DataFrame({'Income': test['Income'], 'True_Label': test['Diabetes_binary'], 'Predicted_Label': y_pred_test})

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


test_accuracies = []

for income_group in income_groups:
    test_income = test[test['Income'] == income_group]
    y_pred_test_income = adaline.predict(test_income.drop('Diabetes_binary', axis=1))

    test_accuracy_income = (y_pred_test_income == test_income['Diabetes_binary']).mean()
    test_accuracies.append(test_accuracy_income)

    
plt.bar(income_groups, test_accuracies)
plt.xlabel('Income Group')
plt.ylabel('Accuracy')
plt.title('Accuracy by Income Group')
plt.show()


### removing 'Education' and retraining

adaline_debiased = AdalineGD(eta = 0.0000001, n_iter = 100000, random_state = 1)
adaline_debiased.fit(train.drop(['Diabetes_binary', 'Education'], axis = 1), train['Diabetes_binary'])

# training F1 and accuracy scores after debiasing
y_pred_train_debiased = adaline_debiased.predict(train.drop(['Diabetes_binary', 'Education'], axis = 1))
train_f1_debiased = f1_score(train['Diabetes_binary'], y_pred_train_debiased)
train_accuracy_debiased = accuracy_score(train['Diabetes_binary'], y_pred_train_debiased)
print(f'Training F1 Score (w/ Debiasing): {train_f1_debiased}')
print(f'Training Accuracy Score (w/ Debiasing): {train_accuracy_debiased}')

# test F1 and accuracy scores after debiasing
y_pred_test_debiased = adaline_debiased.predict(test.drop(['Diabetes_binary', 'Education'], axis = 1))
test_f1_debiased = f1_score(test['Diabetes_binary'], y_pred_test_debiased)
test_accuracy_debiased = accuracy_score(test['Diabetes_binary'], y_pred_test_debiased)
print(f'Testing F1 Score (w/ Debiasing): {test_f1_debiased}')
print(f'Testing Accuracy Score (w/ Debiasing): {test_accuracy_debiased}')

### graphing F1 scores after debiasing

# Calculate the F1 score for each income bracket
df_eval = pd.DataFrame({'Income': test['Income'], 'True_Label': test['Diabetes_binary'], 'Predicted_Label': y_pred_test_debiased})

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
plt.title('False Positives and False Negatives by Income Group (w/ Education Removed)')
plt.xticks([i + bar_width / 2 for i in index], income_groups)
plt.legend()

plt.show()


# In[ ]:




