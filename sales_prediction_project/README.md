# Sales Prediction Project

This project predicts **sales** based on advertising spend and target audience segmentation using Python machine learning.

## Dataset
`sales.csv` contains:
- TV_Spend
- Radio_Spend
- SocialMedia_Spend
- Influencer_Spend
- Target_AgeGroup
- Platform
- Sales (Target)

## Steps to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Predict using the saved model:
```bash
python predict.py
```

## Models Used
- Linear Regression (baseline)
- Random Forest Regressor (final model)

Evaluation Metrics:
- MAE, RMSE, R2 Score
