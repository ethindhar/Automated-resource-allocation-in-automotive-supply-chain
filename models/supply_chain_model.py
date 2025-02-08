import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as XGBRegressor
from datetime import datetime, timedelta

class SupplyChainPredictor:
    def __init__(self):
        self.models = {
            'sales': None,
            'stock': None,
            'utilization': None
        }
        self.scalers = {
            'sales': StandardScaler(),
            'stock': StandardScaler(),
            'utilization': StandardScaler()
        }
        
    def prepare_features(self, df):
        """Prepare features for training"""
        df = df.copy()
        
        # Time-based features
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Rolling features
        windows = [7, 14, 30]
        for window in windows:
            # Sales rolling features
            df[f'SalesMA{window}'] = df.groupby('Region')['SalesVolume'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            df[f'SalesSTD{window}'] = df.groupby('Region')['SalesVolume'].transform(
                lambda x: x.rolling(window, min_periods=1).std())
            
            # Stock rolling features
            df[f'StockMA{window}'] = df.groupby('Region')['CurrentStock'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            
            # Utilization rolling features
            df[f'UtilMA{window}'] = df.groupby('Region')['CurrentUtilization'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
        
        # Cost-related features
        df['TotalCost'] = df['LaborCost'] + df['MaterialCost'] + df['TransportCost']
        df['CostPerUnit'] = df['TotalCost'] / df['SalesVolume'].clip(lower=1)
        
        # Interaction features
        df['StockToSales'] = df['CurrentStock'] / df['SalesVolume'].rolling(7, min_periods=1).mean()
        df['UtilizationEfficiency'] = df['SalesVolume'] / (df['CurrentUtilization'].clip(lower=0.1) * df['CurrentStock'])
        
        # Region encoding
        df = pd.get_dummies(df, columns=['Region'], prefix='Region')
        
        return df
        
    def train_models(self, df):
        """Train models for sales, stock, and utilization prediction"""
        processed_df = self.prepare_features(df)
        
        feature_columns = [col for col in processed_df.columns if col not in 
                         ['Date', 'SalesVolume', 'CurrentStock', 'CurrentUtilization']]
        
        # Train sales model
        X = processed_df[feature_columns]
        y_sales = processed_df['SalesVolume']
        X_scaled = self.scalers['sales'].fit_transform(X)
        
        sales_model = XGBRegressor.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8
        )
        sales_model.fit(X_scaled, y_sales)
        self.models['sales'] = sales_model
        
        # Train stock model
        y_stock = processed_df['CurrentStock']
        X_scaled = self.scalers['stock'].fit_transform(X)
        
        stock_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5
        )
        stock_model.fit(X_scaled, y_stock)
        self.models['stock'] = stock_model
        
        # Train utilization model
        y_util = processed_df['CurrentUtilization']
        X_scaled = self.scalers['utilization'].fit_transform(X)
        
        util_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5
        )
        util_model.fit(X_scaled, y_util)
        self.models['utilization'] = util_model
        
        return {
            'sales_score': sales_model.score(X_scaled, y_sales),
            'stock_score': stock_model.score(X_scaled, y_stock),
            'util_score': util_model.score(X_scaled, y_util)
        }
    
    def predict_future(self, df, days=30):
        """Predict future values for sales, stock, and utilization"""
        last_date = pd.to_datetime(df['Date']).max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        # Create future dataframe with basic features
        future_df = pd.DataFrame({'Date': future_dates})
        last_values = df.iloc[-1].copy()
        
        predictions = []
        
        for date in future_dates:
            # Create a row for prediction
            row = pd.DataFrame([last_values])
            row['Date'] = date
            row = self.prepare_features(row)
            
            # Get feature columns
            feature_columns = [col for col in row.columns if col not in 
                             ['Date', 'SalesVolume', 'CurrentStock', 'CurrentUtilization']]
            
            # Make predictions
            X = row[feature_columns]
            
            sales_pred = self.models['sales'].predict(self.scalers['sales'].transform(X))[0]
            stock_pred = self.models['stock'].predict(self.scalers['stock'].transform(X))[0]
            util_pred = self.models['utilization'].predict(self.scalers['utilization'].transform(X))[0]
            
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'sales': max(0, sales_pred),
                'stock': max(0, stock_pred),
                'utilization': min(1, max(0, util_pred))
            })
            
            # Update last values for next prediction
            last_values['SalesVolume'] = sales_pred
            last_values['CurrentStock'] = stock_pred
            last_values['CurrentUtilization'] = util_pred
        
        return predictions

def train_and_predict(df):
    """Main function to train models and make predictions"""
    try:
        predictor = SupplyChainPredictor()
        scores = predictor.train_models(df)
        predictions = predictor.predict_future(df)
        
        return {
            'status': 'success',
            'data': {
                'model_scores': scores,
                'predictions': predictions
            },
            'message': 'Models trained and predictions generated successfully'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error in training and prediction: {str(e)}',
            'data': None
        } 