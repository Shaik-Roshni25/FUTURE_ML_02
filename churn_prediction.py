import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('churn_data.csv')  # Make sure 'churn_data.csv' is in the same folder as the script

# Data Preprocessing
df.ffill(inplace=True)  # Fill missing values using forward fill

# Encode categorical columns (Contract, InternetService, PaymentMethod)
encoder = LabelEncoder()

# Apply label encoding to each categorical column
df['Contract'] = encoder.fit_transform(df['Contract'])
df['InternetService'] = encoder.fit_transform(df['InternetService'])
df['PaymentMethod'] = encoder.fit_transform(df['PaymentMethod'])

# Encode Gender (already done previously)
df['Gender'] = encoder.fit_transform(df['Gender'])  # Convert 'Male' to 0 and 'Female' to 1

# Separate features and target
X = df.drop(['CustomerID', 'Churn'], axis=1)  # Drop 'CustomerID' and 'Churn' from features
y = df['Churn']  # 'Churn' is the target column

# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model with class imbalance handling
model = xgb.XGBClassifier(scale_pos_weight=3)  # Adjust scale_pos_weight if needed
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
churn_probabilities = model.predict_proba(X_test)[:, 1]  # Get the probability of churn (class 1)

# Save predictions and churn probabilities to a new DataFrame, along with other columns
predictions_df = df.loc[X_test.index, [
    'CustomerID', 'Gender', 'Age', 'Tenure', 'MonthlySpend',
    'Contract', 'InternetService', 'PaymentMethod', 'Churn'
]].copy()

# Add predicted churn and churn probabilities to the DataFrame
predictions_df['Predicted_Churn'] = y_pred
predictions_df['Churn_Probability'] = churn_probabilities

# Save the DataFrame to a CSV file
predictions_df.to_csv('churn_predictions.csv', index=False)  # Save to CSV file

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Feature Importance Plot with error handling
try:
    xgb.plot_importance(model)  # Plot feature importance
    plt.show()  # Display the plot
except ValueError:
    print("Feature importance plot is empty. Model might not have learned anything.")
