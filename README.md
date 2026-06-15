# Wine Quality Prediction Project

## Project Overview
This project aims to build a classification model to predict the quality of red wine based on its physicochemical properties. The goal is to classify wine into 'good quality' (quality >= 7) or 'bad quality' (quality < 7) using various machine learning algorithms.

## Data Source
The dataset used is the Wine Quality dataset (specifically, red wine) from the UCI Machine Learning Repository.

**Source URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv`

## Libraries Used
- `pandas` for data manipulation and analysis.
- `numpy` for numerical operations.
- `matplotlib.pyplot` for basic plotting.
- `seaborn` for advanced data visualization.
- `sklearn.model_selection` for splitting data into training and testing sets.
- `sklearn.metrics` for evaluating model performance (confusion matrix, classification report, accuracy score).
- `sklearn.preprocessing.StandardScaler` for feature scaling (though not used in the final models presented here, it was imported).
- `sklearn.linear_model.LogisticRegression` for Logistic Regression.
- `sklearn.tree.DecisionTreeClassifier` for Decision Tree.
- `sklearn.ensemble.RandomForestClassifier` for Random Forest.

## Project Steps and Analysis

### 1. Data Loading and Initial Inspection
- The `winequality-red.csv` file was downloaded from the specified URL using `requests`.
- The CSV file was loaded into a pandas DataFrame named `wine_df`.
- Initial data inspection was performed using:
    - `wine_df.head()`: Displayed the first few rows.
    - `wine_df.shape`: Showed the number of rows and columns (1599 rows, 12 columns).
    - `wine_df.info()`: Provided a summary of the DataFrame, including data types and non-null counts, confirming no missing values.
    - `wine_df.isnull().sum()`: Explicitly checked for null values, confirming no missing data.
    - `wine_df.describe()`: Generated descriptive statistics for numerical columns.

### 2. Exploratory Data Analysis (EDA)
- **Quality Distribution**: `wine_df['quality'].value_counts()` showed the distribution of wine quality ratings (ranging from 3 to 8), with most wines rated 5 or 6.
- **Visualizations**:
    - `sns.countplot(wine_df['quality'])`: Visualized the distribution of wine quality, reinforcing the imbalance towards quality levels 5 and 6.
    - `wine_df.hist(bins=100, figsize=(10,12))`: Generated histograms for all numerical features to understand their distributions.
    - `sns.heatmap(wine_df.corr(), annot=True)`: Displayed a correlation heatmap to identify relationships between features.
    - `wine_df.corr()['quality'].sort_values()`: Showed features sorted by their correlation with the `quality` target, highlighting `alcohol` and `sulphates` as positively correlated, and `volatile acidity`, `total sulfur dioxide`, and `density` as negatively correlated.
    - *Note on `sns.barplot(wine_df['quality'], wine_df['alcohol'])`*: An initial attempt to use `sns.barplot` directly with two positional arguments resulted in a `TypeError`. This could be corrected by specifying `x` and `y` arguments, e.g., `sns.barplot(x='quality', y='alcohol', data=wine_df)`.

### 3. Data Preprocessing
- **Target Binarization**: The `quality` column was transformed into a binary target variable:
    - Wines with `quality >= 7` were labeled as `1` (good quality).
    - Wines with `quality < 7` were labeled as `0` (bad quality).
    - `wine_df['quality'].value_counts()` after transformation showed 1382 wines labeled 0 and 217 labeled 1, indicating an imbalanced dataset.
- **Feature and Target Split**: The dataset was split into features `X` (all columns except `quality`) and target `y` (`quality` column).
- **Train-Test Split**: The data was further split into training (70%) and testing (30%) sets using `train_test_split` with `random_state=42` for reproducibility.
    - `X_train` shape: (1119, 11)
    - `y_train` shape: (1119,)
    - `X_test` shape: (480, 11)
    - `y_test` shape: (480,)

### 4. Model Training and Evaluation

#### 4.1. Logistic Regression
- A `LogisticRegression` model was trained on the `X_train` and `y_train` data.
- **Test Accuracy**: `86.67%`
- **Classification Report**:
```
              precision    recall  f1-score   support

           0       0.89      0.97      0.93       413
           1       0.56      0.22      0.32        67

    accuracy                           0.87       480
   macro avg       0.72      0.60      0.62       480
weighted avg       0.84      0.87      0.84       480
```
- **Confusion Matrix**:
    - TN: 401, FN: 52, TP: 15, FP: 12

#### 4.2. Decision Tree Classifier
- A `DecisionTreeClassifier` model was trained.
- **Test Accuracy**: `85.42%`
- **Classification Report**:
```
              precision    recall  f1-score   support

           0       0.93      0.89      0.91       413
           1       0.48      0.61      0.54        67

    accuracy                           0.85       480
   macro avg       0.71      0.75      0.73       480
weighted avg       0.87      0.85      0.86       480
```
- **Confusion Matrix**:
    - TN: 369, FN: 26, TP: 41, FP: 44

#### 4.3. Random Forest Classifier
- A `RandomForestClassifier` model was trained.
- **Test Accuracy**: `88.96%`
- **Classification Report**:
```
              precision    recall  f1-score   support

           0       0.92      0.95      0.94       413
           1       0.62      0.52      0.57        67

    accuracy                           0.89       480
   macro avg       0.77      0.74      0.75       480
weighted avg       0.88      0.89      0.89       480
```
- **Confusion Matrix**:
    - TN: 392, FN: 32, TP: 35, FP: 21

## Conclusion
Among the three models tested, the **Random Forest Classifier** achieved the highest test accuracy of `88.96%`. While all models performed reasonably well in classifying the majority class (quality 0), the recall for the minority class (quality 1) remains a challenge across all models due to the imbalanced nature of the dataset. Further steps could involve exploring techniques to handle imbalanced data (e.g., SMOTE, oversampling/undersampling) and hyperparameter tuning to improve the performance, especially for the good quality wines.
