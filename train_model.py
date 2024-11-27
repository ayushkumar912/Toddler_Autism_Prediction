import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  


data = pd.read_csv('Toddler Autism dataset July 2018.csv')

label_encoders = {}
categorical_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

joblib.dump(label_encoders, 'optimized_label_encoders.joblib')


data = data.drop(columns=['Case_No', 'Who completed the test'])

X = data[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Qchat-10-Score', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']]
y = data['Class/ASD Traits'].apply(lambda x: 1 if x == 'Yes' else 0)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


model1 = GaussianNB()
model1.fit(X_train_scaled, y_train)


model2 = RandomForestClassifier( n_estimators=30,  
        max_depth=7, 
        min_samples_leaf=6,   
        random_state=42)
model2.fit(X_train_scaled, y_train)


ensemble_model = VotingClassifier(estimators=[('nb', model1), ('rf', model2)], voting='soft',weights=[1,0.6])
ensemble_model.fit(X_train_scaled, y_train)

joblib.dump(model1, 'gaussian_nb_model.joblib')
joblib.dump(model2, 'random_forest_model.joblib')
joblib.dump(ensemble_model, 'ensemble_model.joblib')

proba_model1 = model1.predict_proba(X_test_scaled)
proba_model2 = model2.predict_proba(X_test_scaled)
proba_ensemble = ensemble_model.predict_proba(X_test_scaled)

print("Predicted probabilities using Naive Bayes:")
print(proba_model1)

print("\nPredicted probabilities using Random Forest:")
print(proba_model2)

print("\nPredicted probabilities using Ensemble model:")
print(proba_ensemble)

accuracy_nb = accuracy_score(y_test, model1.predict(X_test_scaled))
accuracy_rf = accuracy_score(y_test, model2.predict(X_test_scaled))
accuracy_ensemble = accuracy_score(y_test, ensemble_model.predict(X_test_scaled))

print(f"\nAccuracy of Naive Bayes: {accuracy_nb:.2f}")
print(f"Accuracy of Random Forest: {accuracy_rf:.2f}")
print(f"Accuracy of Ensemble model: {accuracy_ensemble:.2f}")

print("\nClassification Report for Naive Bayes:")
print(classification_report(y_test, model1.predict(X_test_scaled)))

print("Classification Report for Random Forest:")
print(classification_report(y_test, model2.predict(X_test_scaled)))

print("Classification Report for Ensemble model:")
print(classification_report(y_test, ensemble_model.predict(X_test_scaled)))

print("Confusion Matrix for Naive Bayes:")
print(confusion_matrix(y_test, model1.predict(X_test_scaled)))

print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, model2.predict(X_test_scaled)))

print("Confusion Matrix for Ensemble model:")
print(confusion_matrix(y_test, ensemble_model.predict(X_test_scaled)))
