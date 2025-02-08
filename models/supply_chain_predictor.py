import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class SupplyChainPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for the model"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Time-based features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        
        # Rolling averages
        df['SalesMA7'] = df['SalesVolume'].rolling(window=7, min_periods=1).mean()
        df['SalesMA30'] = df['SalesVolume'].rolling(window=30, min_periods=1).mean()
        
        # Stock and utilization features
        df['StockToSales'] = df['CurrentStock'] / (df['SalesVolume'].rolling(window=7, min_periods=1).mean() + 1)
        df['UtilizationRate'] = df['CurrentUtilization']
        
        # Fill any missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        return df
    
    def train_models(self, df):
        """Train the prediction model"""
        try:
            processed_df = self.prepare_features(df)
            
            features = [
                'DayOfWeek', 'Month', 'Quarter', 'Year',
                'SalesMA7', 'SalesMA30', 'StockToSales', 
                'UtilizationRate', 'CurrentStock', 'CurrentUtilization'
            ]
            
            X = processed_df[features]
            y = processed_df['SalesVolume']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            
            # Calculate model score
            score = self.model.score(X_scaled, y)
            
            return {
                'model_score': score,
                'features_used': features
            }
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return None
    
    def predict_future(self, df, days=30):
        """Generate predictions for future days"""
        try:
            if self.model is None:
                raise Exception("Model not trained. Please train the model first.")
            
            last_date = pd.to_datetime(df['Date']).max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
            
            # Create future dataframe
            future_df = pd.DataFrame()
            future_df['Date'] = future_dates
            
            # Copy last values for initial predictions
            last_values = df.iloc[-1].copy()
            
            predictions = []
            current_stock = float(last_values['CurrentStock'])
            current_utilization = float(last_values['CurrentUtilization'])
            
            for date in future_dates:
                # Prepare features for prediction
                pred_df = pd.DataFrame([{
                    'Date': date,
                    'CurrentStock': current_stock,
                    'CurrentUtilization': current_utilization,
                    'DayOfWeek': date.dayofweek,
                    'Month': date.month,
                    'Quarter': (date.month - 1) // 3 + 1,
                    'Year': date.year,
                    'SalesMA7': df['SalesVolume'].tail(7).mean(),
                    'SalesMA30': df['SalesVolume'].tail(30).mean(),
                    'StockToSales': current_stock / (df['SalesVolume'].tail(7).mean() + 1),
                    'UtilizationRate': current_utilization
                }])
                
                # Scale features
                X_pred = self.scaler.transform(pred_df[self.model.feature_names_in_])
                
                # Make prediction
                sales_pred = float(self.model.predict(X_pred)[0])
                
                # Update stock and utilization based on prediction
                current_stock = max(0, current_stock - sales_pred)
                current_utilization = min(1, max(0, current_utilization + np.random.normal(0, 0.05)))
                
                predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'sales': max(0, sales_pred),
                    'stock': current_stock,
                    'utilization': current_utilization
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None 