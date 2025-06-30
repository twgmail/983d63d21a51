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

3. **Model Training and Evaluation**:
   - Added option for hyperparameter tuning with GridSearchCV
   - Added ROC AUC score as an additional evaluation metric
   - Added feature importance analysis

4. **Path Handling**:
   - Simplified path handling for better maintainability

5. **Error Handling**:
   - Added robust error handling for file operations and data processing

## Usage

Run the script with:

```bash
python example-0.py
```

To enable hyperparameter tuning, set `perform_grid_search=True` in the `main()` function.