# -*- coding: utf-8 -*-
'''
created on sun dec 1 12:05:14 2024

@author: Irene Chau
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler  # StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE  

# step 1: load dataframe
df = pd.read_csv('file.csv')  # replace 'file.csv' with dataset

# --- step 2: data cleaning ---
# handle missing values
print('initial data overview:')
print(df.info())  # overview of missing data and column types

print("Finding nulls per column")
df.isnull().any()

# drop rows or columns with too many missing values
df = df.dropna(thresh=len(df) * 0.5, axis=1)  # drop columns where >50% values are missing
df = df.dropna(inplace=True)  # drop rows with missing values (or use imputation if critical)

# handle duplicate rows
df = df.drop_duplicates()

# --- step 6: visualize outliers (before removing them) ---
plt.figure(figsize=(14, 8))
sns.boxplot(df, x='col name'))
plt.title('Box plot of numerical columns to identify outliers')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.show()

# --- step 6.1: replace outliers (if known) ---
# example: replace outliers in a numerical column with mean/median
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr  # change the threshold/multiplier if needed
    upper_bound = q3 + 1.5 * iqr
    
    # Print calculated bounds
    print(f'{column}: Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr:.2f}')
    print(f'Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}')
    
    # Identify and count outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    inliers = len(df) - outliers.shape[0]
    print(f'Amount of outliers in {column}: {outliers.shape[0]:,}, Amount of inliers: {inliers:,}')
    
    # Replace outliers with the median
    df[column] = np.where(
        (df[column] < lower_bound) | (df[column] > upper_bound),
        df[column].median(),
        df[column]
    )

# --- step 3: define target variable(s) ---
# identify and isolate the target variable
target_column = 'target'  # replace 'target' with your actual target column name
if target_column not in df.columns:
    raise ValueError(f'target column {target_column} not found in dataset.')
    
# separate features and target
X = df.drop(target_column, axis=1)  # features
y = df[target_column]  # target

# --- step 4: remove unnecessary rows and columns ---
# drop uninformative columns
drop_columns = ['id', 'timestamp']  # replace with column names that are irrelevant to your analysis
df = df.drop(columns=[col for col in drop_columns if col in df.columns])

# filter unnecessary rows
# example: remove rows with zero or irrelevant values in a key column
key_column = 'important_metric'  # replace with your key column name
if key_column in df.columns:
    df = df[df[key_column] > 0]

# --- step 5: data reformatting and restructuring ---
# convert data types (if needed)
# example: convert categorical variables to 'category' dtype
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].astype('category')

# example: convert datetime columns
datetime_columns = ['date_column']  # replace with your datetime column names
for col in datetime_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# encode categorical variables (one-hot encoding or label encoding)
df = pd.get_dummies(df, drop_first=True)  # one-hot encode all categorical variables

# --- step 7: check for missing values ---
print('missing values overview:')
print(df.isnull().sum())  # prints the count of missing values for each column

# --- step 8: normalization ---
# normalize numerical columns
scaler = MinMaxScaler()  # or use StandardScaler
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# step 8.1: split the data into train and test sets first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# step 8.2: fit the scaler on the training data only and transform both training and test data
scaler.fit(X_train[numerical_columns])
X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# --- step 9: handle imbalanced data ---
# use SMOTE for oversampling minority classes (classification problems only)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# --- step 10: train/test split or k-folds ---
# train/test split (already done in step 8)
# k-folds cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_resampled, y_resampled):
    X_train_fold, X_test_fold = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train_fold, y_test_fold = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
    # train and validate your model inside this loop

print('data preprocessing complete and ready for modeling!')
