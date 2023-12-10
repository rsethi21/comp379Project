from imblearn.over_sampling import RandomOverSampler
import pandas as pd


data = pd.read_csv('diabetes.csv')
# Separate data by income levels
income_levels = data['Income'].unique()
balanced_data = []

for level in income_levels:
    subset = data[data['Income'] == level]
    
    print(f"Income level: {level}")
    print(f"Class distribution before balancing:\n{subset['Diabetes_binary'].value_counts()}")
    # Check class imbalance
    class_counts = subset['Diabetes_binary'].value_counts()
    minority_class = class_counts.idxmin()
    minority_class_count = class_counts.min()
    majority_class_count = class_counts.max()

    if minority_class_count < majority_class_count:
        # Apply Random Oversampling to balance the minority class
        features = subset.drop('Diabetes_binary', axis=1)  # Exclude the 'Diabetes_binary' column
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        features_resampled, labels_resampled = ros.fit_resample(features, subset['Diabetes_binary'])
        
        # Combine resampled data with original data
        resampled_df = pd.DataFrame(features_resampled, columns=features.columns)
        resampled_df['Diabetes_binary'] = labels_resampled
        balanced_data.append(resampled_df)

        print(f"Class distribution after balancing:\n{resampled_df['Diabetes_binary'].value_counts()}")
    else:
        # If class is already balanced, keep the original data
        balanced_data.append(subset)
        print("Class distribution is already balanced.")
        print(f"Class distribution:\n{subset['Diabetes_binary'].value_counts()}")

# Combine balanced data for all income levels
balanced_data = pd.concat(balanced_data, axis=0)

balanced_data.to_csv('balanced_data.csv', index=False)

# Now 'balanced_data' contains the balanced dataset for each income level
