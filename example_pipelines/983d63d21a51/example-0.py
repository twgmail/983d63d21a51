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


def preprocess_data_initial(data):
    """
    Initial preprocessing before train/test split.

    Args:
        data (pandas.DataFrame): Raw dataset

    Returns:
        tuple: (X, y) initial features and target
    """
    # Create a copy to avoid modifying the original data
    data_copy = data.copy()

    # Extract deck information from Cabin before dropping
    data_copy["Deck"] = data_copy["Cabin"].str[0].fillna("U")  # U for Unknown

    # Select features and target
    X = data_copy.drop(["Survived", "Name", "Ticket", "Cabin", "PassengerId"], axis=1)
    y = data_copy["Survived"]

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded


def preprocess_features(X_train, X_test):
    """
    Preprocess features after train/test split to avoid data leakage.

    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Test features

    Returns:
        tuple: (X_train_processed, X_test_processed) preprocessed features
    """
    # Create copies to avoid modifying original data
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    # Handle missing values in categorical columns using training set statistics
    if "Embarked" in X_train_copy.columns and X_train_copy["Embarked"].isnull().any():
        embarked_mode = X_train_copy["Embarked"].mode()[0]
        X_train_copy["Embarked"] = X_train_copy["Embarked"].fillna(embarked_mode)
        X_test_copy["Embarked"] = X_test_copy["Embarked"].fillna(embarked_mode)

    # One-hot encode categorical variables
    # Apply to training set first to establish the feature set
    X_train_encoded = pd.get_dummies(X_train_copy, columns=["Sex", "Embarked", "Deck"])

    # For test set, we need to ensure it has exactly the same columns as training set
    X_test_encoded = pd.get_dummies(X_test_copy, columns=["Sex", "Embarked", "Deck"])

    # Get the training set columns (this is our reference feature set)
    train_columns = X_train_encoded.columns.tolist()

    # Ensure test set conforms exactly to training set feature structure
    # Add missing columns (that exist in training but not in test) with zeros
    for col in train_columns:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0

    # Remove extra columns (that exist in test but not in training)
    # and reorder to match training set exactly
    X_test_encoded = X_test_encoded[train_columns]

    # Handle missing values in numerical columns using training set statistics
    if "Age" in X_train_encoded.columns and X_train_encoded["Age"].isnull().any():
        age_mean = X_train_encoded["Age"].mean()
        X_train_encoded["Age"] = X_train_encoded["Age"].fillna(age_mean)
        X_test_encoded["Age"] = X_test_encoded["Age"].fillna(age_mean)

    if "Fare" in X_train_encoded.columns and X_train_encoded["Fare"].isnull().any():
        fare_mean = X_train_encoded["Fare"].mean()
        X_train_encoded["Fare"] = X_train_encoded["Fare"].fillna(fare_mean)
        X_test_encoded["Fare"] = X_test_encoded["Fare"].fillna(fare_mean)

    # Print preprocessing summary
    print(f"Preprocessing complete. Training features shape: {X_train_encoded.shape}")
    print(f"Test features shape: {X_test_encoded.shape}")
    print(f"Missing values in training set: {X_train_encoded.isnull().sum().sum()}")
    print(f"Missing values in test set: {X_test_encoded.isnull().sum().sum()}")

    return X_train_encoded, X_test_encoded


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
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
        }

        print("Starting grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,  # Reduced from 5 to 3 for faster execution
            scoring="accuracy",
            n_jobs=-1,  # Use all available cores
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
                "accuracy": accuracy,
                "classification_report": report,
                "auc": auc,
            }
        else:
            print(
                "Warning: Cannot calculate ROC AUC score with only one class in test set"
            )
            metrics = {"accuracy": accuracy, "classification_report": report}
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC score: {e}")
        metrics = {"accuracy": accuracy, "classification_report": report}

    return metrics


def main():
    """Main function to run the ML pipeline."""
    try:
        # Simpler path handling
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        raw_data_file = os.path.join(
            project_root, "datasets", "8c2a25260209", "8c2a25260209.csv"
        )

        print(f"Starting ML pipeline with dataset: {raw_data_file}")

        # Load data
        data = load_data(raw_data_file)
        print(f"Data loaded successfully. Shape: {data.shape}")

        # Initial preprocessing (before train/test split)
        X, y = preprocess_data_initial(data)

        # Split data BEFORE final preprocessing to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(
            f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets"
        )

        # Preprocess features using only training set statistics
        X_train_processed, X_test_processed = preprocess_features(X_train, X_test)

        # Train model (set perform_grid_search=True to enable hyperparameter tuning)
        print("Training model...")
        model = train_model(X_train_processed, y_train, perform_grid_search=False)

        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test_processed, y_test)

        # Print results
        print("\n----- Model Evaluation Results -----")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Classification report:\n{metrics['classification_report']}")
        if "auc" in metrics:
            print(f"ROC AUC: {metrics['auc']}")

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "Feature": X_train_processed.columns,
                "Importance": model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        print("\n----- Feature Importance -----")
        print(feature_importance.head(10))

        print("\nML pipeline completed successfully!")

    except Exception as e:
        print(f"Error in ML pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
