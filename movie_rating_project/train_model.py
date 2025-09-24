import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "movies.csv")

# Basic preprocessing
df = df.dropna(subset=["Rating"])
# Keep a copy
df["Actors"] = df["Actors"].fillna("Unknown")
df["Director"] = df["Director"].fillna("Unknown")
df["Genre"] = df["Genre"].fillna("Unknown")

# Feature engineering:
# - One-hot encode Genre (low cardinality)
# - Vectorize cast (director + actors) with CountVectorizer (sparse)
df["cast_director"] = df["Director"] + " | " + df["Actors"]

# Numeric features
num_cols = ["Budget", "BoxOffice", "Runtime"]

# Categorical: Genre
cat_cols = ["Genre"]

# Text: cast_director
text_col = "cast_director"

# Build transformers
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
cv = CountVectorizer(max_features=200)  # pick top 200 tokens

# Prepare matrices
X_num = df[num_cols].fillna(0).values
X_cat = ohe.fit_transform(df[cat_cols])
X_text = cv.fit_transform(df[text_col].fillna("")).toarray()

import numpy as np
X = np.hstack([X_num, X_cat, X_text])
y = df["Rating"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))

# Save model components and model
joblib.dump({
    "model": model,
    "ohe": ohe,
    "cv": cv,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "text_col": text_col
}, BASE / "movie_rating_model.joblib")

print("Saved model to movie_rating_model.joblib")