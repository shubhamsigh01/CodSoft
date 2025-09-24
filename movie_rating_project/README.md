# Movie Rating Prediction — Sample Project

This small project demonstrates how to build a regression model to predict movie ratings
from features like genre, director, actors, budget, box office and runtime.

**Files included**:
- `movies.csv` — synthetic sample dataset (300 rows)
- `train_model.py` — script to preprocess data, train a model, evaluate and save it
- `predict.py` — example script that loads the saved model and predicts a rating for a sample movie
- `preprocess.py` — helper functions for encoding / feature engineering
- `requirements.txt` — Python dependencies
- `README.md` — this file

## Quick start (run locally)

1. Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate    # windows
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Train model:
```
python train_model.py
```

4. Predict with saved model:
```
python predict.py
```

The `train_model.py` script will save a model file `movie_rating_model.joblib` in the project folder.