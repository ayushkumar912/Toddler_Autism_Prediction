import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class DualBayesAutismModel:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.gaussian_nb = GaussianNB()
        self.bernoulli_nb = BernoulliNB()
        self.voting_classifier = None
        
    def preprocess_data(self, data):
        """Preprocess the dataset with essential feature engineering."""
        df = data.copy()
        
    
        df = df.fillna(df.mode().iloc[0])
        
        df['Behavioral_Score'] = df[['A1', 'A2', 'A3', 'A4', 'A5', 
                                   'A6', 'A7', 'A8', 'A9', 'A10']].sum(axis=1)

        categorical_columns = [
            'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD',
            'Who completed the test', 'Class/ASD Traits'
        ]
        
        for column in categorical_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            self.label_encoders[column] = le
        
        return df
    
    def prepare_features(self, df):
        """Prepare and split features for different model types."""
        
        gaussian_features = [
            'Age_Mons', 'Qchat-10-Score', 'Behavioral_Score'
        ]
        
       
        bernoulli_features = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
            'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD',
            'Who completed the test'
        ]

        df[gaussian_features] = self.scaler.fit_transform(df[gaussian_features])
        X = df[gaussian_features + bernoulli_features]
        y = df['Class/ASD Traits']
        
        return X, y
    
    def train_model(self, X, y):
        """Train both models and create a voting classifier."""

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        self.voting_classifier = VotingClassifier(
            estimators=[
                ('gaussian_nb', self.gaussian_nb),
                ('bernoulli_nb', self.bernoulli_nb)
            ],
            voting='soft'
        )
        
        self.voting_classifier.fit(X_train, y_train)

        y_pred = self.voting_classifier.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print("\nModel Performance:")
        print("=" * 50)
        print("\nVoting Classifier Performance:")
        print(classification_report(y_test, y_pred))
        
        cv_scores = cross_val_score(self.voting_classifier, X_resampled, y_resampled, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return metrics, (X_test, y_test)
    
    def save_model(self, path_prefix='optimized_'):
        """Save the trained model and associated transformers."""
        joblib.dump(self.voting_classifier, f'{path_prefix}voting_classifier.joblib')
        joblib.dump(self.label_encoders, f'{path_prefix}label_encoders.joblib')
        joblib.dump(self.scaler, f'{path_prefix}scaler.joblib')
        print("Model and associated objects saved successfully.")
    
    def predict_single_case(self, case_data):
        """Predict for a single case."""

        processed_case = self.preprocess_data(pd.DataFrame([case_data]))
        features, _ = self.prepare_features(processed_case)
        
        prediction = self.voting_classifier.predict(features)
        prediction_proba = self.voting_classifier.predict_proba(features)
        
        return prediction[0], prediction_proba[0]


if __name__ == "__main__":
    
    print("Loading dataset...")
    data = pd.read_csv("Toddler Autism dataset July 2018.csv")
    
    model = DualBayesAutismModel()
    
    print("Preprocessing data...")
    processed_data = model.preprocess_data(data)
    
    print("Preparing features...")
    X, y = model.prepare_features(processed_data)
    
    print("Training and evaluating model...")
    metrics, test_data = model.train_model(X, y)
    
    model.save_model()

    print("\nFinal Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.3f}")