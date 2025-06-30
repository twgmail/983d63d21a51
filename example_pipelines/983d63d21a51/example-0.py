import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

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
    # Create a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # Extract deck information from Cabin before dropping
    data_copy['Deck'] = data_copy['Cabin'].str[0].fillna('U')  # U for Unknown
    
    # Select features and target
    X = data_copy.drop(["Survived", "Name", "Ticket", "Cabin", "PassengerId"], axis=1)
    y = data_copy["Survived"]
    
    # Handle missing values in categorical columns
    if "Embarked" in X.columns and X["Embarked"].isnull().any():
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=["Sex", "Embarked", "Deck"])
    
    # Handle missing values in numerical columns
    if "Age" in X.columns and X["Age"].isnull().any():
        X["Age"] = X["Age"].fillna(X["Age"].mean())
    
    if "Fare" in X.columns and X["Fare"].isnull().any():
        X["Fare"] = X["Fare"].fillna(X["Fare"].mean())
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Print preprocessing summary
    print(f"Preprocessing complete. Features shape: {X.shape}")
    print(f"Missing values after preprocessing: {X.isnull().sum().sum()}")
    
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
        # Use a smaller parameter grid for faster execution
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        
        print("Starting grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42), 
            param_grid=param_grid, 
            cv=3,  # Reduced from 5 to 3 for faster execution
            scoring='accuracy',
            n_jobs=-1  # Use all available cores
        )
        
        try:
            grid_search.fit(X_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            print(f"Grid search failed: {e}. Falling back to default model.")
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)
            return clf
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
        # ROC AUC score requires at least two classes
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_pred_proba)
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'auc': auc
            }
        else:
            print("Warning: Cannot calculate ROC AUC score with only one class in test set")
            metrics = {
                'accuracy': accuracy,
                'classification_report': report
            }
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC score: {e}")
        metrics = {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    return metrics

def main():
    """Main function to run the ML pipeline."""
    try:
        # Simpler path handling
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        raw_data_file = os.path.join(project_root, "datasets", "8c2a25260209", "8c2a25260209.csv")
        
        print(f"Starting ML pipeline with dataset: {raw_data_file}")
        
        # Load data
        data = load_data(raw_data_file)
        print(f"Data loaded successfully. Shape: {data.shape}")
        
        # Preprocess data
        X, y = preprocess_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets")
        
        # Train model (set perform_grid_search=True to enable hyperparameter tuning)
        print("Training model...")
        model = train_model(X_train, y_train, perform_grid_search=False)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Print results
        print("\n----- Model Evaluation Results -----")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Classification report:\n{metrics['classification_report']}")
        if 'auc' in metrics:
            print(f"ROC AUC: {metrics['auc']}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n----- Feature Importance -----")
        print(feature_importance.head(10))
        
        print("\nML pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in ML pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
