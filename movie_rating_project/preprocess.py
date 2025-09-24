import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnJoiner(BaseEstimator, TransformerMixin):
    """Join director and actors into a single 'cast_director' feature (string)."""
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xc = X.copy()
        Xc["cast_director"] = Xc[self.cols].agg(" | ".join, axis=1)
        return Xc[["cast_director"]]

def build_preprocessor():
    # We'll one-hot encode Genre and extract a simple frequency encoding for cast_director
    from sklearn.feature_extraction.text import CountVectorizer
    # For simplicity use CountVectorizer on the combined cast_director string to produce sparse features
    def identity(x): return x
    return None  # handled in scripts explicitly