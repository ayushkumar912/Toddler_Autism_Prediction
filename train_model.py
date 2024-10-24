import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # Importing joblib for saving models
import numpy as np

# Load the dataset (assuming CSV format)
data = pd.read_csv('Toddler Autism dataset July 2018.csv')

# Preprocessing: Label encoding for categorical columns
label_encoders = {}
categorical_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Save label encoders to a file
joblib.dump(label_encoders, 'optimized_label_encoders.joblib')

# Drop unnecessary columns
data = data.drop(columns=['Case_No', 'Who completed the test'])

# Features (X) and labels (y)
X = data[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Qchat-10-Score', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']]
y = data['Class/ASD Traits'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data into train+validation (80%) and test (20%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split train+validation into train (70%) and validation (30%) sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Model 1: Naive Bayes
model1 = GaussianNB()
model1.fit(X_train_scaled, y_train)

# Train Model 2: Random Forest
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train_scaled, y_train)

# Ensemble: Voting Classifier (soft voting)
ensemble_model = VotingClassifier(estimators=[('nb', model1), ('rf', model2)], voting='soft')
ensemble_model.fit(X_train_scaled, y_train)

# # Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')  # Save the scaler

# Save the models using joblib
joblib.dump(model1, 'gaussian_nb_model.joblib')
joblib.dump(model2, 'random_forest_model.joblib')
joblib.dump(ensemble_model, 'ensemble_model.joblib')

# Predict probabilities on the test set
proba_model1 = model1.predict_proba(X_test_scaled)
proba_model2 = model2.predict_proba(X_test_scaled)
proba_ensemble = ensemble_model.predict_proba(X_test_scaled)

# Print the predicted probabilities for the test set
print("Predicted probabilities using Naive Bayes:")
print(proba_model1)

print("\nPredicted probabilities using Random Forest:")
print(proba_model2)

print("\nPredicted probabilities using Ensemble model:")
print(proba_ensemble)

# Optional: Evaluate the models (accuracy, classification report, confusion matrix)
accuracy_nb = accuracy_score(y_test, model1.predict(X_test_scaled))
accuracy_rf = accuracy_score(y_test, model2.predict(X_test_scaled))
accuracy_ensemble = accuracy_score(y_test, ensemble_model.predict(X_test_scaled))

print(f"\nAccuracy of Naive Bayes: {accuracy_nb:.2f}")
print(f"Accuracy of Random Forest: {accuracy_rf:.2f}")
print(f"Accuracy of Ensemble model: {accuracy_ensemble:.2f}")

# Print classification reports
print("\nClassification Report for Naive Bayes:")
print(classification_report(y_test, model1.predict(X_test_scaled)))

print("Classification Report for Random Forest:")
print(classification_report(y_test, model2.predict(X_test_scaled)))

print("Classification Report for Ensemble model:")
print(classification_report(y_test, ensemble_model.predict(X_test_scaled)))

# Print confusion matrices
print("Confusion Matrix for Naive Bayes:")
print(confusion_matrix(y_test, model1.predict(X_test_scaled)))

print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, model2.predict(X_test_scaled)))

print("Confusion Matrix for Ensemble model:")
print(confusion_matrix(y_test, ensemble_model.predict(X_test_scaled)))
