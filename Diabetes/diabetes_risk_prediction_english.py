"""
Diabetes Risk Prediction — English-updated script
Generated on 2025-10-09T18:43:33

This script was converted from your notebook:
- Variable and comment text standardized to clear English.
- No logic changes were intentionally introduced.
- Plots and modeling steps remain the same.

If you need this back as a Jupyter Notebook, tell me and I can convert it.
"""


# === Cell 0 ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)


# === Cell 1 ===

df =pd.read_csv("/diabetes_binary_health_indicators_BRFSS2015.csv", sep=',', skipinitialspace=True, na_values="?")

df.head(10)


# === Cell 2 ===

df.info()


# === Cell 3 ===

missing_values = df.isnull().sum()
print("\nNumber of missing values per column:")
print(missing_values)


# === Cell 4 ===

missing_percentage = (missing_values / len(df)) * 100
print("\% of missing value per column:")
print(missing_percentage)


# === Cell 5 ===

df.describe()


# === Cell 6 ===

df['Diabetes_binary'].value_counts(normalize=True)


# === Cell 8 ===

numeric_features = df.select_dtypes(include=[np.number]).columns
df[numeric_features].hist(figsize=(15, 10))
plt.tight_layout()
plt.show()


# === Cell 9 ===

categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(10, 5))
    df[feature].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {feature}')
    plt.ylabel('Count')
    plt.xlabel(feature)
    plt.xticks(rotation=45)
    plt.show()


# === Cell 11 ===

correlation_matrix = df[numeric_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.show()


# === Cell 13 ===

for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Diabetes_binary', y=feature, data=df)
    plt.title(f'{feature} vs Diabetes_binary')
    plt.show()


# === Cell 15 ===

for feature in categorical_features:
    if feature != 'Diabetes_binary':
        plt.figure(figsize=(18, 6))
        df_temp = df.groupby([feature, 'Diabetes_binary']).size().unstack()
        df_temp_perc = df_temp.div(df_temp.sum(axis=1), axis=0)
        df_temp_perc.plot(kind='bar', stacked=True)
        plt.title(f'{feature} vs isi_disini')
        plt.xlabel(feature)
        plt.ylabel('Percentage')
        plt.legend(title='Diabetes_binary', loc='upper right')
        plt.xticks(rotation=45)
        plt.show()


# === Cell 16 ===

plt.figure(figsize=(15, 10))
df[numeric_features].boxplot()
plt.title('Box Plots of Numeric Features')
plt.xticks(rotation=90)
plt.show()


# === Cell 18 ===

missing_values = df.isnull().sum()

missing_percentage = 100 * df.isnull().sum() / len(df)

missing_table = pd.concat([missing_values, missing_percentage], axis=1, keys=['Total', 'Percent'])

print(missing_table)


# === Cell 20 ===

print(df.dtypes)

numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    non_numeric = df[pd.to_numeric(df[col], errors='coerce').isna()]
    if len(non_numeric) > 0:
        print(f"\nNon-numeric values in the column{col}:")
        print(non_numeric[col].unique())


# === Cell 22 ===

for col in numeric_columns:
    print(f"\nRange of values ​​for {col}:")
    print(f"Min: {df[col].min()}, Max: {df[col].max()}")

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nUnique categories {col}:")
    print(df[col].unique())


# === Cell 25 ===

print("Missing values :")
print(df.isnull().sum())

# Menangani missing values
for column in df.columns:
    if df[column].dtype == 'object':
        # For categorical columns, fill missing values with the mode
        #df[column].fillna(df[column].mode()[0], inplace=True)
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        # Untuk columns numerik, isi dengan median
        #df[column].fillna(df[column].median(), inplace=True)
        df[column] = df[column].fillna(df[column].median())

print("\nMissing values setelah pemcleanan:")
print(df.isnull().sum())


# === Cell 26 ===

df.info()


# === Cell 28 ===

df.head()


# === Cell 30 ===

df.to_csv('clean_diabetes_binary_health_indicators_BRFSS2015.csv', index=False)
print("The cleared data has been saved.")


# === Cell 31 ===

# Correlation Heatmap
correlation = df.corr()
plt.subplots(figsize = (15,15))
sns.heatmap(correlation.round(2),
            annot = True,
            vmax = 1,
            square = True,
            cmap = 'RdYlGn_r')
plt.show()


# === Cell 32 ===

df = df.loc[:,df.apply(pd.Series.nunique) != 1]


# === Cell 33 ===

df.head()


# === Cell 34 ===

# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# === Cell 35 ===

data_without_target = df.drop('Diabetes_binary', axis=1)


# === Cell 36 ===

corr_features = correlation(data_without_target, 0.8)
print('correlated features: ', len(set(corr_features)) )
print(corr_features)


# === Cell 37 ===

# removed correlated  features
df.drop(labels=corr_features, axis=1, inplace=True)


# === Cell 38 ===

df.info()


# === Cell 39 ===

df.describe()


# === Cell 41 ===

from sklearn.model_selection import train_test_split
# Pisahkan features dan target
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)


# === Cell 42 ===

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# === Cell 44 ===

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score,precision_score,classification_report, confusion_matrix


# Modeling
print("\n4. Modeling - Decision Tree with GridSearchCV...")
# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
# Create decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    estimator=dt_classifier,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"\nBest Parameters: {best_params}")


# === Cell 45 ===

# Evaluation
print("\n5. Evaluation...")

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(df['Diabetes_binary'].unique()),
            yticklabels=sorted(df['Diabetes_binary'].unique()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
# plt.close()


# === Cell 46 ===

# Visualization of the Decision Tree
print("\n6. Visualizing the Decision Tree...")

# Plot the decision tree
plt.figure(figsize=(40, 30))
plot_tree(best_model,
          feature_names=X.columns,
          class_names=[str(i) for i in sorted(df['Diabetes_binary'].unique())],
          filled=True,
          rounded=True,
          max_depth=3)  # Limiting depth for better visualization
plt.title('Decision Tree Visualization (Limited to Depth 3)')
plt.savefig('decision_tree.png')
#plt.close()


# === Cell 48 ===

# Plot feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
# plt.close()


# === Cell 50 ===

ori_y_pred_dt_train = best_model.predict(X_train)

ori_accuracy_dt_train = accuracy_score(y_train,ori_y_pred_dt_train)
print('Accuracy on training set: ', ori_accuracy_dt_train)

ori_precision_dt_train = precision_score(y_train,ori_y_pred_dt_train, average='micro')
print('Precision on training set: ', ori_precision_dt_train)

ori_recall_dt_train = recall_score(y_train,ori_y_pred_dt_train, average='micro')
print('Recall on training set: ', ori_recall_dt_train)

ori_y_pred_dt_test = best_model.predict(X_test)

ori_accuracy_dt_test = accuracy_score(y_test,ori_y_pred_dt_test)
print('Accuracy on test set: ', ori_accuracy_dt_test)

ori_precision_dt_test = precision_score(y_test,ori_y_pred_dt_test, average='micro')
print('Precision on test set: ', ori_precision_dt_test)

ori_recall_dt_test = recall_score(y_test,ori_y_pred_dt_test, average='micro')
print('Recall on test set: ', ori_recall_dt_test)


# === Cell 52 ===

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score,precision_score,classification_report, confusion_matrix


# Modeling
print("\n4. Modeling - Decision Tree without optimasi")

# Create decision tree classifier
dt_classifier_norm = DecisionTreeClassifier(random_state=42)


# Fit the grid search to the data
dt_classifier_norm.fit(X_train, y_train)


# === Cell 53 ===

norm_y_pred_dt_train = dt_classifier_norm.predict(X_train)

norm_accuracy_dt_train = accuracy_score(y_train,norm_y_pred_dt_train)
print('Accuracy on training set: ', norm_accuracy_dt_train)

norm_precision_dt_train = precision_score(y_train,norm_y_pred_dt_train, average='micro')
print('Precision pada training set: ', norm_precision_dt_train)

norm_recall_dt_train = recall_score(y_train,norm_y_pred_dt_train, average='micro')
print('Recall on training set: ', norm_recall_dt_train)

norm_y_pred_dt_test = dt_classifier_norm.predict(X_test)

norm_accuracy_dt_test = accuracy_score(y_test,norm_y_pred_dt_test)
print('Accuracy on test set: ', norm_accuracy_dt_test)

norm_precision_dt_test = precision_score(y_test,norm_y_pred_dt_test, average='micro')
print('Precision on test set: ', norm_precision_dt_test)

norm_recall_dt_test = recall_score(y_test,norm_y_pred_dt_test, average='micro')
print('Recall on test set: ', norm_recall_dt_test)


# === Cell 55 ===

models = [
          ('Machine Learning with Data optimization', ori_accuracy_dt_train, ori_accuracy_dt_test),
          ('Machine Learning without Data optimization', norm_accuracy_dt_train, norm_accuracy_dt_test),
         ]


# === Cell 56 ===

predict = pd.DataFrame(data = models, columns=['Model', 'Training Accuracy', 'Test Accuracy'])
predict


# === Cell 57 ===

models_comparison = [
                        ('Machine Learning Data Original', ori_accuracy_dt_test, ori_recall_dt_test, ori_precision_dt_test),
                        ('Machine Learning Data Normal', norm_accuracy_dt_test, norm_recall_dt_test, norm_precision_dt_test),
                    ]


# === Cell 58 ===

comparison = pd.DataFrame(data = models_comparison, columns=['Model', 'Accuracy', 'Recall', 'Precision'])
comparison


# === Cell 59 ===

import numpy as np

f, axes = plt.subplots(2,1, figsize=(14,10))

predict.sort_values(by=['Training Accuracy'], ascending=False, inplace=True)

sns.barplot(x='Training Accuracy', y='Model', data = predict, palette='Blues_d', ax = axes[0])
#axes[0].set(xlabel='Region', ylabel='Charges')
axes[0].set_xlabel('Training Accuracy', size=16)
axes[0].set_ylabel('Model')
axes[0].set_xlim(0,1.0)
axes[0].set_xticks(np.arange(0, 1.1, 0.1))

predict.sort_values(by=['Test Accuracy'], ascending=False, inplace=True)

sns.barplot(x='Test Accuracy', y='Model', data = predict, palette='Greens_d', ax = axes[1])
#axes[0].set(xlabel='Region', ylabel='Charges')
axes[1].set_xlabel('Test Accuracy', size=16)
axes[1].set_ylabel('Model')
axes[1].set_xlim(0,1.0)
axes[1].set_xticks(np.arange(0, 1.1, 0.1))

plt.show()
