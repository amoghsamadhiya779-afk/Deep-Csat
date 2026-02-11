import joblib
import pandas as pd
import numpy as np
import os
import logging

class CSATInference:
    def __init__(self, model_dir='models', model_name='csat_model.pkl'):
        self.model_path = os.path.join(model_dir, model_name)
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.load_model()

    def load_model(self):
        """Loads the trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Model loaded from {self.model_path}")
        else:
            self.logger.error(f"Model not found at {self.model_path}. Please train the model first.")
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    def predict(self, data):
        """
        Accepts a dictionary or DataFrame and returns predictions.
        """
        # Ensure data is a DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Input must be a dictionary or pandas DataFrame")

        # The pipeline inside the model handles all preprocessing (filling NaNs, encoding, etc.)
        # We just need to ensure the columns match the training data
        
        # If the model pipeline expects 'response_time_minutes' but it's missing,
        # we ensure it exists and fill it with np.nan (standard float missing value)
        if 'response_time_minutes' not in df.columns:
            df['response_time_minutes'] = np.nan  # <--- FIXED: Changed pd.NA to np.nan

        # Ensure numeric columns are actually numeric (handle strings like "300")
        numeric_cols = ['Item_price', 'connected_handling_time', 'response_time_minutes']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        try:
            prediction = self.model.predict(df)
            probability = self.model.predict_proba(df).max(axis=1) if hasattr(self.model, "predict_proba") else None
            return prediction, probability
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise