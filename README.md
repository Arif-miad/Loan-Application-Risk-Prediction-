
<div style="background-color: #f4f4f4; color: #333; border-radius: 15px; margin: 20px; padding: 30px; text-align: center; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
  <h1 style="color: #2C3E50;">Welcome to My Project! 🚀</h1>
  <p style="font-size: 18px; line-height: 1.6;">
    This project is about <strong>Loan Application Risk Prediction</strong> using machine learning.
  </p>
  
  <h3 style="color: #16A085;">Connect with Me 👨‍💻</h3>
  <p style="font-size: 16px;">
    📧 <a href="mailto:arifmiahcse@gmail.com" style="color: #1abc9c; text-decoration: none;">arifmiahcse@gmail.com</a><br>
    📈 <a href="https://www.kaggle.com/arifmia" target="_blank" style="color: #1abc9c; text-decoration: none;">My Kaggle Profile</a><br>
    💼 <a href="https://www.linkedin.com/in/arif-miah-8751bb217" target="_blank" style="color: #1abc9c; text-decoration: none;">My LinkedIn Profile</a>
  </p>

  <h3 style="color: #16A085;">Skills & Expertise 💻</h3>
  <p>
    🔹 Machine Learning<br>
    🔹 Data Science<br>
    🔹 Deep Learning<br>
    🔹 Python Programming<br>
    🔹 Computer Vision<br>
    🔹 Kaggle Notebooks Expert
  </p>

  <h3 style="color: #16A085;">Follow Me for Updates 🔔</h3>
  <p>Stay tuned for more exciting projects and updates!</p>
</div>

---

# Loan Application Risk Prediction 📊💳

## Project Overview 🚀

This project focuses on predicting the risk of loan applications using machine learning techniques. The goal is to predict whether a loan application will be approved or rejected based on various features like age of the applicant, loan amount, employment type, loan purpose, and more.

The dataset contains historical loan application data with details of applicants and their loan statuses. By analyzing these features, we aim to build a robust machine learning model that can classify loan applications as **success** (approved) or **failure** (rejected).

## Dataset Information 📑

The dataset consists of two main files:

- **loan_applications.csv**: Contains the loan application details along with success/rejection labels.
- **credit_features_subset.csv**: Includes additional credit-related features for applicants.
- **loan_data_dictionary.csv**: A data dictionary explaining the features of the dataset.

### Columns in the dataset:
- **UID**: Unique identifier for each applicant.
- **AgeOfOldestAccount**: Age of the oldest account.
- **AgeOfYoungestAccount**: Age of the youngest account.
- **Count**: Number of accounts.
- **CountActive**: Number of active accounts.
- **CountClosedLast12Months**: Number of accounts closed in the last 12 months.
- **CountDefaultAccounts**: Number of default accounts.
- **CountOpenedLast12Months**: Number of accounts opened in the last 12 months.
- **CountSettled**: Number of settled accounts.
- **MeanAccountAge**: Average age of the applicant's accounts.
- **SumCurrentOutstandingBal**: Total outstanding balance on the applicant's accounts.
- **TimeSinceMostRecentDefault**: Time since the applicant’s most recent default.
- **WorstPaymentStatusActiveAccounts**: Worst payment status on active accounts.
- **Amount**: Loan amount requested by the applicant.
- **Term**: Repayment period requested.
- **EmploymentType**: Employment status of the applicant.
- **LoanPurpose**: Purpose for the loan.
- **Success**: Target variable (0 = Rejected, 1 = Approved).

## Objectives 🎯

- **Data Preprocessing**: Handle missing values, outliers, and scale features for better model performance.
- **Exploratory Data Analysis (EDA)**: Visualize the dataset to identify patterns and correlations between features and the target variable.
- **Model Building**: Train different classification models to predict the loan application status.
- **Model Evaluation**: Evaluate models using appropriate metrics like accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.

## Code Implementation 🧑‍💻

This repository contains Python code that implements the following steps:

1. **Data Preprocessing**:
   - Handling missing values
   - Feature scaling and normalization
   - Encoding categorical features using Label Encoding
   - Removing outliers

2. **Exploratory Data Analysis (EDA)**:
   - Visualizing numerical and categorical features
   - Analyzing relationships between the target variable (`Success`) and other features

3. **Model Training**:
   - Building various classification models including Logistic Regression, Random Forest, SVM, and others.
   - Hyperparameter tuning for better model performance.

4. **Model Evaluation**:
   - Evaluating models using confusion matrix, precision, recall, F1-score, and ROC-AUC curve.

5. **Feature Engineering**:
   - Creating new features that could enhance the prediction model.

## Usage Instructions 📌

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/loan-application-risk-prediction.git
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook or Python script to preprocess the data, build models, and evaluate them:
   ```bash
   jupyter notebook loan_application_analysis.ipynb
   ```

## Libraries Used 📚

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Evaluation Metrics 📈

- **Accuracy**: Measures the overall accuracy of the model.
- **Precision**: Measures the number of correct positive predictions out of all positive predictions.
- **Recall**: Measures the number of correct positive predictions out of all actual positive cases.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Measures the area under the ROC curve to evaluate classifier performance.

## Results and Insights 💡

- After performing model evaluation, the **Random Forest** model outperformed others in terms of accuracy and ROC-AUC score, indicating it as the most reliable model for loan risk prediction.
- Feature analysis showed that **loan amount** and **employment type** were strong predictors of loan approval.

## Contributing 🤝

Feel free to fork this project, open issues, and submit pull requests. Contributions are always welcome!

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.







## Workflow 🔄

### 1. **Data Preprocessing** 🧹

In this step, the raw data is cleaned and prepared for machine learning models.

- **Missing Values**: Handle missing values using appropriate strategies (e.g., mean imputation for numerical features).
- **Outlier Removal**: Identify and remove outliers using **box plots** and **IQR (Interquartile Range)**.
- **Scaling and Normalization**: Scale numerical features using **MinMaxScaler** or **StandardScaler** to ensure all features are on the same scale.
- **Encoding Categorical Variables**: Convert categorical variables to numerical using **Label Encoding**.

```python
# Handle missing values
df.fillna(df.mean(), inplace=True)

# Remove outliers using IQR
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numerical_columns] < (Q1 - 1.5 * IQR)) | (df[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Scaling features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Label encoding for categorical features
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df[categorical_columns] = df[categorical_columns].apply(encoder.fit_transform)
```

### 2. **Exploratory Data Analysis (EDA)** 🔍

This step helps us understand the data by visualizing relationships between features and the target variable.

- **Visualize the distribution** of numerical and categorical features.
- **Correlation Analysis**: Examine how features are correlated with the target variable.
- **Class Distribution**: Analyze the distribution of the target variable (`Success`).

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of numerical columns
df[numerical_columns].hist(bins=20, figsize=(15,10))
plt.show()

# Boxplot to detect outliers
sns.boxplot(data=df[numerical_columns])
plt.show()

# Correlation heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# Class distribution of target variable
sns.countplot(x='Success', data=df)
plt.show()
```

### 3. **Model Building** 🤖

In this phase, different machine learning models are trained and evaluated to predict the target variable.

- **Train models**: Logistic Regression, Random Forest, Support Vector Machine (SVM), and XGBoost.
- **Hyperparameter Tuning**: Use techniques like **GridSearchCV** or **RandomizedSearchCV** for better model performance.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Split the dataset into training and testing sets
X = df.drop('Success', axis=1)
y = df['Success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Train the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate model performance
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.show()
```

### 4. **Model Evaluation** 📊

We evaluate the performance of the model using various metrics such as **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and the **Confusion Matrix**.

- **Confusion Matrix**: Shows the number of true positives, false positives, true negatives, and false negatives.
- **ROC-AUC Curve**: Provides an overall measure of the model's performance.

```python
# Plot the ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

### 5. **Model Comparison** ⚖️

You can compare different models to identify which one performs the best.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier()
}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    print("-" * 50)
```


## Conclusion 🎯

- The **Random Forest** model showed the best results in terms of **accuracy**, **precision**, **recall**, and **ROC-AUC**.
- Feature importance analysis indicated that the **loan amount** and **employment type** are strong predictors of loan approval.

---



