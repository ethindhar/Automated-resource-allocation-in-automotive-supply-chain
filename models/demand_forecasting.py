import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def prepare_features(df):
    """Enhanced feature engineering with NaN handling"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Time-based features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Advanced rolling statistics with minimum periods
    for window in [7, 14, 30]:
        df[f'SalesMA{window}'] = df['SalesVolume'].rolling(window=window, min_periods=1).mean()
        df[f'SalesSTD{window}'] = df['SalesVolume'].rolling(window=window, min_periods=1).std()
        df[f'SalesMin{window}'] = df['SalesVolume'].rolling(window=window, min_periods=1).min()
        df[f'SalesMax{window}'] = df['SalesVolume'].rolling(window=window, min_periods=1).max()
    
    # Lag features with forward fill for NaN values
    for lag in [1, 7, 14]:
        df[f'SalesLag{lag}'] = df['SalesVolume'].shift(lag)
        df[f'SalesLag{lag}'] = df[f'SalesLag{lag}'].fillna(method='ffill')
        df[f'SalesLag{lag}'] = df[f'SalesLag{lag}'].fillna(method='bfill')
    
    # Interaction features with safe division
    df['StockToSales'] = df['CurrentStock'] / (df['SalesVolume'].rolling(window=7, min_periods=1).mean() + 1)
    df['UtilizationRate'] = df['CurrentUtilization']
    df['StockUtilizationInteraction'] = df['CurrentStock'] * df['CurrentUtilization']
    
    # Fill any remaining NaN values with appropriate methods
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    return df

def train_forecast_model(df):
    """Enhanced model training with NaN handling"""
    features = [
        'DayOfWeek', 'Month', 'Quarter', 'WeekOfYear', 'IsWeekend',
        'SalesMA7', 'SalesMA14', 'SalesMA30',
        'SalesSTD7', 'SalesSTD14', 'SalesSTD30',
        'SalesMin7', 'SalesMax7',
        'SalesLag1', 'SalesLag7', 'SalesLag14',
        'StockToSales', 'UtilizationRate', 'StockUtilizationInteraction'
    ]
    
    X = df[features]
    y = df['SalesVolume']
    
    # Handle any remaining NaN values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=features)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model with optimized parameters
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    )
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Calculate metrics
    test_pred = model.predict(X_test)
    
    metrics = {
        'r2_score': r2_score(y_test, test_pred),
        'mae': mean_absolute_error(y_test, test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'feature_importance': dict(zip(features, model.feature_importances_))
    }
    
    return model, scaler, imputer, metrics, features

def prepare_prediction_data(input_values, days_to_predict):
    """Prepare data for prediction"""
    current_date = datetime.strptime(input_values['date'], '%Y-%m-%d')
    future_dates = pd.date_range(
        start=current_date + timedelta(days=1), 
        periods=days_to_predict, 
        freq='D'
    )
    
    # Initialize prediction data
    pred_data = []
    current_stock = float(input_values['stock'])
    current_utilization = float(input_values['utilization'])
    last_sales = float(input_values['sales'])
    
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
            'CostPerUnit': float(input_values['cost']),
            'UtilizationEfficiency': last_sales / (current_utilization * current_stock + 0.1),
            'SeasonalityIndex': 1.0,
            # Region encoding
            'Region_Asia': 1 if input_values['region'] == 'Asia' else 0,
            'Region_Europe': 1 if input_values['region'] == 'Europe' else 0,
            'Region_North America': 1 if input_values['region'] == 'North America' else 0
        }
        pred_data.append(row)
    
    return pd.DataFrame(pred_data)

def predict_demand(input_values):
    """Predict demand using the trained model"""
    try:
        # Load model and scaler
        model = joblib.load('supply_chain_model.joblib')
        scaler = joblib.load('feature_scaler.joblib')
        
        # Get days_to_predict from input_values
        days_to_predict = int(input_values['days'])
        
        # Prepare prediction data
        pred_df = prepare_prediction_data(input_values, days_to_predict)
        
        # Select features in correct order
        features = [
            'DayOfWeek', 'Month', 'Quarter', 'IsWeekend',
            'SalesMA7', 'SalesMA30', 'SalesSTD7',
            'StockMA7', 'UtilMA7',
            'StockToSales', 'CostPerUnit',
            'UtilizationEfficiency', 'SeasonalityIndex',
            'Region_Asia', 'Region_Europe', 'Region_North America'
        ]
        
        X_pred = scaler.transform(pred_df[features])
        predictions = model.predict(X_pred)
        
        # Calculate confidence intervals (example using standard deviation)
        prediction_std = np.std(predictions) * 2  # 95% confidence interval
        lower_bound = predictions - prediction_std
        upper_bound = predictions + prediction_std
        
        return {
            'dates': pred_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'predictions': predictions.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'current_metrics': {
                'current_stock': float(input_values['stock']),
                'current_utilization': float(input_values['utilization']),
                'last_actual_sales': float(input_values['sales'])
            }
        }
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")
