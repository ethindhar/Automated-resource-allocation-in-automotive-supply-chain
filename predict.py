import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_user_input():
    """Get input values from user"""
    print("\n=== Supply Chain Prediction System ===")
    print("\nPlease enter the current values:")
    
    try:
        # Get date input
        while True:
            date_str = input("\nEnter current date (YYYY-MM-DD): ")
            try:
                current_date = datetime.strptime(date_str, '%Y-%m-%d')
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD format.")
        
        # Get numeric inputs with validation
        current_stock = float(input("Enter current stock level: "))
        while current_stock < 0:
            print("Stock cannot be negative. Please enter a valid number.")
            current_stock = float(input("Enter current stock level: "))
        
        current_utilization = float(input("Enter current utilization rate (0-1): "))
        while not 0 <= current_utilization <= 1:
            print("Utilization must be between 0 and 1.")
            current_utilization = float(input("Enter current utilization rate (0-1): "))
        
        current_sales = float(input("Enter current sales volume: "))
        while current_sales < 0:
            print("Sales cannot be negative. Please enter a valid number.")
            current_sales = float(input("Enter current sales volume: "))
        
        cost_per_unit = float(input("Enter cost per unit: "))
        while cost_per_unit < 0:
            print("Cost cannot be negative. Please enter a valid number.")
            cost_per_unit = float(input("Enter cost per unit: "))
        
        # Get region input
        print("\nSelect region:")
        print("1. North")
        print("2. South")
        print("3. East")
        print("4. West")
        
        region_choice = int(input("Enter region number (1-4): "))
        while region_choice not in [1, 2, 3, 4]:
            print("Invalid choice. Please select 1-4.")
            region_choice = int(input("Enter region number (1-4): "))
        
        # Create region encoding
        regions = ['North', 'South', 'East', 'West']
        region_encoding = {f'Region_{region}': 1 if i+1 == region_choice else 0 
                         for i, region in enumerate(regions)}
        
        # Days to predict
        days_to_predict = int(input("\nEnter number of days to predict: "))
        while days_to_predict <= 0:
            print("Number of days must be positive.")
            days_to_predict = int(input("Enter number of days to predict: "))
        
        # Combine all inputs
        input_data = {
            'Date': current_date,
            'CurrentStock': current_stock,
            'CurrentUtilization': current_utilization,
            'SalesVolume': current_sales,
            'CostPerUnit': cost_per_unit,
            **region_encoding
        }
        
        return input_data, days_to_predict
        
    except ValueError as e:
        print(f"\nError: Please enter valid numeric values. {str(e)}")
        return None, None

def prepare_prediction_data(input_values, days_to_predict):
    """Prepare data for prediction"""
    # Create future dates
    future_dates = pd.date_range(
        start=input_values['Date'] + timedelta(days=1), 
        periods=days_to_predict, 
        freq='D'
    )
    
    # Initialize prediction data
    pred_data = []
    current_stock = float(input_values['CurrentStock'])
    current_utilization = float(input_values['CurrentUtilization'])
    last_sales = float(input_values['SalesVolume'])
    
    for date in future_dates:
        row = {
            'Date': date,
            'DayOfWeek': date.dayofweek,
            'Month': date.month,
            'Quarter': (date.month - 1) // 3 + 1,
            'IsWeekend': int(date.dayofweek in [5, 6]),
            'SalesMA7': last_sales,
            'SalesMA30': last_sales,
            'SalesSTD7': 0,
            'StockMA7': current_stock,
            'UtilMA7': current_utilization,
            'StockToSales': current_stock / (last_sales + 1),
            'CostPerUnit': input_values['CostPerUnit'],
            'UtilizationEfficiency': last_sales / (current_utilization * current_stock + 0.1),
            'SeasonalityIndex': 1.0
        }
        
        # Add region encoding
        for col, value in input_values.items():
            if col.startswith('Region_'):
                row[col] = value
        
        pred_data.append(row)
    
    return pd.DataFrame(pred_data)

def make_predictions():
    """Main prediction function"""
    try:
        # Get user input
        input_values, days_to_predict = get_user_input()
        if input_values is None:
            return
        
        print("\nLoading model and making predictions...")
        
        # Load the trained model and scaler
        model = joblib.load('supply_chain_model.joblib')
        scaler = joblib.load('feature_scaler.joblib')
        
        # Prepare prediction data
        pred_df = prepare_prediction_data(input_values, days_to_predict)
        
        # Select features
        features = [
            'DayOfWeek', 'Month', 'Quarter', 'IsWeekend',
            'SalesMA7', 'SalesMA30', 'SalesSTD7',
            'StockMA7', 'UtilMA7',
            'StockToSales', 'CostPerUnit',
            'UtilizationEfficiency', 'SeasonalityIndex'
        ] + [col for col in pred_df.columns if col.startswith('Region_')]
        
        # Scale features and predict
        X_pred = scaler.transform(pred_df[features])
        predictions = model.predict(X_pred)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': pred_df['Date'].dt.strftime('%Y-%m-%d'),
            'Predicted_Sales': np.round(predictions, 2),
            'Predicted_Stock': np.round(pred_df['StockMA7'], 2),
            'Predicted_Utilization': np.round(pred_df['UtilMA7'], 3)
        })
        
        # Display results
        print("\n=== Prediction Results ===")
        print(results.to_string(index=False))
        
        # Save predictions
        filename = f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results.to_csv(filename, index=False)
        print(f"\nPredictions saved to '{filename}'")
        
        # Display summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Average Predicted Sales: {predictions.mean():.2f}")
        print(f"Maximum Predicted Sales: {predictions.max():.2f}")
        print(f"Minimum Predicted Sales: {predictions.min():.2f}")
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        print("Please make sure the model files exist and inputs are correct.")

if __name__ == "__main__":
    make_predictions() 