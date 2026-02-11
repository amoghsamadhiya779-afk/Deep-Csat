import logging
import os
from src.csat_pipelining import CSATPredictor

# Configure global logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # Define paths relative to the root directory
    DATA_PATH = os.path.join("data", "eCommerce_Customer_support_data.csv")
    MODEL_DIR = "models"

    print("------------------------------------------------")
    print("      DeepCSAT Pipeline Execution Setup")
    print("------------------------------------------------")

    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at '{DATA_PATH}'.")
        print("Please run 'python create_structure.py' to reorganize your files,")
        print("or ensure the CSV is placed inside the 'data' folder.")
        return

    # Initialize and run predictor
    predictor = CSATPredictor(data_path=DATA_PATH, model_dir=MODEL_DIR)
    predictor.run()

if __name__ == "__main__":
    main()