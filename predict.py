from src.inference import CSATInference
import pandas as pd

def main():
    print("------------------------------------------------")
    print("      DeepCSAT Inference Test")
    print("------------------------------------------------")

    # 1. Initialize Inference Engine
    try:
        engine = CSATInference()
    except FileNotFoundError:
        print("Error: Model not found. Run 'python main.py' to train it first.")
        return

    # 2. Define Sample Data
    # FIXED: Added 'Product_category' which was causing the error
    sample_customer = {
        'channel_name': 'Inbound',
        'category': 'Product Queries',
        'Sub-category': 'Product Specific Information',
        'Product_category': 'Electronics',  # <--- This was missing!
        'Customer Remarks': 'The agent was very helpful and solved my issue quickly.',
        'Item_price': 150.00,
        'connected_handling_time': 300,
        'Tenure Bucket': 'On Job Training',
        'Agent Shift': 'Morning',
        'Manager': 'John Doe'
    }

    print(f"\nProcessing Sample Data:\n{sample_customer}")

    # 3. Make Prediction
    try:
        prediction, proba = engine.predict(sample_customer)
        
        print("\n------------------------------------------------")
        print(f"Predicted CSAT Score: {prediction[0]}")
        if proba is not None:
            print(f"Confidence: {proba[0]:.2%}")
        print("------------------------------------------------")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()