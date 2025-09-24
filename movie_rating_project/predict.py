import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
obj = joblib.load(BASE / "movie_rating_model.joblib")

model = obj["model"]
ohe = obj["ohe"]
cv = obj["cv"]
num_cols = obj["num_cols"]
cat_cols = obj["cat_cols"]
text_col = obj["text_col"]

# Example new movie
sample = {
    "Title": "New Blockbuster",
    "Genre": "Action",
    "Director": "Director_1",
    "Actors": "Actor_1, Actor_2, Actor_3",
    "Budget": 80000000,
    "BoxOffice": 200000000,
    "Runtime": 125
}

df = pd.DataFrame([sample])
df["cast_director"] = df["Director"] + " | " + df["Actors"]

X_num = df[num_cols].fillna(0).values
X_cat = ohe.transform(df[cat_cols])
X_text = cv.transform(df[text_col].fillna("")).toarray()

X = np.hstack([X_num, X_cat, X_text])
pred = model.predict(X)
print(f"Predicted rating for '{sample['Title']}': {pred[0]:.2f}")