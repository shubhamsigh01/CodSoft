# Credit Card Fraud Detection — Sample Project

This project demonstrates how to build a machine learning model to detect fraudulent credit card transactions.

It uses a synthetic dataset with imbalanced classes (fraud vs. genuine) and applies oversampling with SMOTE to balance data.

**Files included**:
- `creditcard.csv` — synthetic dataset (5000 transactions, ~2% fraud)
- `train_model.py` — preprocess, balance data, train Logistic Regression and Random Forest, evaluate
- `requirements.txt` — dependencies
- `README.md` — instructions

## Quick start

1. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate    # windows
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Train & evaluate:
```
python train_model.py
```

This prints classification reports and metrics (precision, recall, F1-score, ROC-AUC).