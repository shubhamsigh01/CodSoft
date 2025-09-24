import pandas as pd
import joblib

# Load trained model
model = joblib.load("sales_model.pkl")

# Example new data for prediction
new_data = pd.DataFrame({
    "TV_Spend": [10000],
    "Radio_Spend": [5000],
    "SocialMedia_Spend": [8000],
    "Influencer_Spend": [3000],
    "Target_AgeGroup": [2],  # Example: encoded value for '36-50'
    "Platform": [1]          # Example: encoded value for 'Offline'
})

# Predict sales
prediction = model.predict(new_data)
print("Predicted Sales:", prediction[0])
