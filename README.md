# email-spam-detector-ml
> Repository: email-spam-detector-ml
A machine learning project for spam email detection, using Logistic Regression (73.79% accuracy) and Random Forest to classify normal/spam emails with 100 features.

## Project Overview
- Task: Binary classification of normal vs. spam emails
- Data Scale: 600K training samples + 540K test samples
- Features: 100 continuous features (e.g., sending frequency, content metrics)
- Best Model: Logistic Regression (73.79% validation accuracy)
- Tech Stack: Python 3.x, pandas, numpy, scikit-learn, matplotlib, seaborn

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas>=1.5.0 numpy>=1.21.0 scikit-learn>=1.2.0 matplotlib>=3.7.0 seaborn>=0.12.0 scipy>=1.9.0 jupyter>=1.0.0
```

### 2. Data Preparation
Data Source: The dataset is available on Kaggle. If you need the data, please visit Kaggle to download the required files (train.csv, test.csv, sample_submission.csv).
- `train.csv`: 600K samples (id + 100 features + target)
- `test.csv`: 540K samples (id + 100 features)
- `sample_submission.csv`: Submission format example

### 3. Run the Project
```bash
# Via Jupyter Notebook
jupyter notebook spam_email_classification.ipynb
```

### 4. Output
- `my_submission.csv`: Test set predictions (id + target)
- Visualization charts: Label distribution, feature correlation, model comparison, etc.

## Core Workflow
1. Data Loading: Validate and load datasets
2. Data Quality Check: No missing/duplicate values, balanced labels (49.4% normal / 50.6% spam)
3. Preprocessing: Outlier handling, Z-score standardization, stratified train-validation split (80%/20%)
4. Model Training: Compare Logistic Regression and Random Forest
5. Prediction: Generate test set results and save

## Key Results
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 73.79%   | 0.74      | 0.75   | 0.74     |
| Random Forest       | 66.50%   | 0.67      | 0.66   | 0.66     |

### Top Predictive Features
1. `f34` (correlation: 0.135)
2. `f55` (correlation: -0.114)
3. `f43` (correlation: 0.109)
4. `f71` (correlation: -0.108)
5. `f80` (correlation: -0.107)

## Project Structure
```
├── spam_email_classification.ipynb  # Main notebook
├── train.csv                        # Training dataset
├── test.csv                         # Test dataset
├── sample_submission.csv            # Submission format example
├── my_submission.csv                # Prediction results
└── README.md                        # Documentation
```

## Usage Example
```python
# Full workflow execution
if __name__ == "__main__":
    # Load data
    train_df, test_df, sample_sub = load_data()
    
    # Preprocess data
    X_train_split, X_val_split, y_train_split, y_val_split, X_test_scaled, test_ids, feature_cols = preprocess_data(train_df, test_df)
    
    # Train models
    final_model, model_name = train_evaluate_models(X_train_split, X_val_split, y_train_split, y_val_split)
    
    # Predict and save
    submission_df = predict_test_set(final_model, X_test_scaled, test_ids)
    print("All processes completed!")
```

## Future Improvements
- Feature selection and hyperparameter tuning
- Try advanced models (XGBoost, LightGBM)
- Integrate email content text analysis for multi-modal classification

---
Star if you find this project useful! For issues, open an issue or submit a PR.
