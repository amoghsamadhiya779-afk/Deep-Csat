import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
import os

class CSATPredictor:
    def __init__(self, data_path, model_dir='models'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'csat_model.pkl')
        self.model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Configure logging specifically for this module
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def load_data(self):
        """Loads data from CSV."""
        self.logger.info(f"Loading data from {self.data_path}...")
        try:
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            self.logger.error(f"File not found at {self.data_path}. Please check the 'data' folder.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def feature_engineering(self, df):
        """
        Performs custom feature engineering.
        """
        self.logger.info("Starting feature engineering...")
        
        # 1. Date Time Conversions
        date_cols = ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # 2. Calculate Response Time (in minutes)
        if 'issue_responded' in df.columns and 'Issue_reported at' in df.columns:
            df['response_time_minutes'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60.0
        else:
            df['response_time_minutes'] = np.nan

        # 3. Clean Item Price
        if 'Item_price' in df.columns:
            df['Item_price'] = pd.to_numeric(df['Item_price'], errors='coerce')

        # 4. Clean Text Data (Fill NaNs here to avoid pipeline dimension issues)
        if 'Customer Remarks' in df.columns:
            df['Customer Remarks'] = df['Customer Remarks'].fillna('').astype(str)

        self.logger.info("Feature engineering completed.")
        return df

    def build_pipeline(self):
        """
        Constructs the sklearn preprocessing and modeling pipeline.
        """
        numeric_features = ['Item_price', 'connected_handling_time', 'response_time_minutes']
        categorical_features = ['channel_name', 'category', 'Sub-category', 'Product_category', 'Tenure Bucket', 'Agent Shift', 'Manager']
        # Note: Passing this as a string ensures TfidfVectorizer receives a Series (1D), which it expects.
        text_features = 'Customer Remarks'

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # REMOVED SimpleImputer from here. We handled NaNs in feature_engineering.
        # TfidfVectorizer works directly on the pandas Series.
        text_transformer = TfidfVectorizer(max_features=100, stop_words='english')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('txt', text_transformer, text_features)
            ],
            remainder='drop'
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])

        return pipeline

    def run(self):
        """
        Executes the full pipeline: Load -> Process -> Train -> Evaluate.
        """
        df = self.load_data()
        df = self.feature_engineering(df)

        target_col = 'CSAT Score'
        
        # Drop rows where target is missing
        df = df.dropna(subset=[target_col])
        
        X = df
        y = df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.logger.info("Building model...")
        self.model = self.build_pipeline()
        
        self.logger.info("Training model...")
        self.model.fit(self.X_train, self.y_train)

        self.logger.info("Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        
        acc = accuracy_score(self.y_test, y_pred)
        self.logger.info(f"Model Accuracy: {acc:.4f}")
        print("\nClassification Report:\n" + classification_report(self.y_test, y_pred))

        # Save model to the models/ directory
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        self.logger.info(f"Model saved to '{self.model_path}'")