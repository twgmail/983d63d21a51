# Improved Machine Learning Pipeline

This directory contains an improved machine learning pipeline for the Titanic dataset.

## Improvements Made

The original script (`example-0.py`) has been refactored with the following improvements:

1. **Code Organization**:
   - Restructured code into modular functions with proper docstrings
   - Added a main function to orchestrate the pipeline
   - Improved error handling with try-except blocks

2. **Data Preprocessing**:
   - Added handling for missing values in all columns (Age, Fare)
   - Extracted deck information from Cabin before dropping it
   - Added checks for missing values in categorical columns
   - Added data validation to ensure columns exist before processing
   - Created a copy of the original data to avoid modifying it
   - **Fixed data leakage**: Split preprocessing into two phases to ensure imputation statistics are calculated only from training data

3. **Model Training and Evaluation**:
   - Added option for hyperparameter tuning with GridSearchCV
   - Added ROC AUC score as an additional evaluation metric
   - Added feature importance analysis
   - Added proper error handling for edge cases (e.g., single class in test set)
   - Used stratified sampling to maintain class distribution in train/test split

4. **Path Handling**:
   - Simplified path handling for better maintainability

5. **Error Handling**:
   - Added robust error handling for file operations and data processing
   - Added informative error messages and warnings
   - Added global try-except in main function to catch and report any errors

6. **Usability Improvements**:
   - Added progress messages to track pipeline execution
   - Added summary statistics for preprocessing and model performance
   - Improved formatting of output for better readability

## Usage

Run the script with:

```bash
python example-0.py
```

To enable hyperparameter tuning, set `perform_grid_search=True` in the `main()` function. Note that grid search may take longer to run but can result in better model performance.

## Data Leakage Fix

**Critical Issue Identified and Fixed**: The original code had a data leakage problem where imputation statistics (mean for Age/Fare, mode for Embarked) were calculated on the entire dataset before splitting into train/test sets. This gave the model indirect access to test set information during training.

**Solution**: Split preprocessing into two phases:
1. `preprocess_data_initial()`: Basic preprocessing before train/test split
2. `preprocess_features()`: Calculate imputation statistics only from training data and apply to both train/test sets

This ensures the test set remains truly unseen during preprocessing and training.

## Performance

The improved pipeline maintains the same core functionality while adding robustness, better error handling, and more informative output. The code is now more maintainable, follows ML best practices, and can handle a wider range of edge cases without data leakage.