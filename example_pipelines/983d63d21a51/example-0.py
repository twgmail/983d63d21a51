import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load and return the dataset from the given file path.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data):
    """
    Preprocess the data for model training.
    
    Args:
        data (pandas.DataFrame): Raw dataset
        
    Returns:
        tuple: (X, y) preprocessed features and target
    """
    # Extract deck information from Cabin before dropping
    data['Deck'] = data['Cabin'].str[0].fillna('U')  # U for Unknown
    
    # Select features and target
    X = data.drop(["Survived", "Name", "Ticket", "Cabin", "PassengerId"], axis=1)
    y = data["Survived"]
    
    # Handle missing values in categorical columns
    if X["Embarked"].isnull().any():
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=["Sex", "Embarked", "Deck"])
    
    # Handle missing values in numerical columns
    X["Age"] = X["Age"].fillna(X["Age"].mean())
    X["Fare"] = X["Fare"].fillna(X["Fare"].mean())
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded

def train_model(X_train, y_train, perform_grid_search=False):
    """
    Train a RandomForestClassifier model.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (array): Training target
        perform_grid_search (bool): Whether to perform grid search for hyperparameter tuning
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model
    """
    if perform_grid_search:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42), 
            param_grid=param_grid, 
            cv=5, 
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        return clf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test (pandas.DataFrame): Test features
        y_test (array): Test target
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'auc': auc
        }
    except:
        metrics = {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    return metrics

def main():
    """Main function to run the ML pipeline."""
    # Simpler path handling
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_data_file = os.path.join(project_root, "datasets", "8c2a25260209", "8c2a25260209.csv")
    
    # Load data
    data = load_data(raw_data_file)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model (set perform_grid_search=True to enable hyperparameter tuning)
    model = train_model(X_train, y_train, perform_grid_search=False)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print results
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Classification report:\n{metrics['classification_report']}")
    if 'auc' in metrics:
        print(f"ROC AUC: {metrics['auc']}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()
