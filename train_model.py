import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib  # For saving the model

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("Toddler Autism dataset July 2018.csv")

# Encoding categorical variables
print("Encoding categorical variables...")
label_encoders = {}
categorical_columns = [
    'Sex', 
    'Ethnicity', 
    'Jaundice', 
    'Family_mem_with_ASD', 
    'Who completed the test', 
    'Class/ASD Traits'
]

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
print("Encoding complete.\n")

# Prepare the features and target variable
X = data[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Qchat-10-Score', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test']]
y = data['Class/ASD Traits']

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples\n")

# Initialize Naive Bayes Classifier
print("Initializing Naive Bayes Classifier...")
model = GaussianNB()

# Train the model
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.\n")

# Make predictions on the test set
print("Making predictions on the test set...")
y_pred = model.predict(X_test)

# Calculate evaluation metrics
print("\nCalculating evaluation metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Display predictions
for i in range(len(y_test)):
    print(f"Sample {i+1}: Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")

# Output model performance metrics
print("\n===== Model Performance on Test Data =====")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("==========================================\n")

# Save the trained model using joblib
joblib.dump(model, 'naive_bayes_model.joblib')

# Save the label encoders for later use in predictions
joblib.dump(label_encoders, 'label_encoders.joblib')

print("Model and label encoders saved successfully.")
