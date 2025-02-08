import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib

# 1. Load and Prepare Data
def load_and_augment_data(file_path):
    """Load and augment the supply chain dataset"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Time-based features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Rolling statistics
    windows = [7, 14, 30]
    for window in windows:
        # Sales patterns
        df[f'SalesMA{window}'] = df.groupby('Region')['SalesVolume'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'SalesSTD{window}'] = df.groupby('Region')['SalesVolume'].transform(
            lambda x: x.rolling(window, min_periods=1).std())
        
        # Stock patterns
        df[f'StockMA{window}'] = df.groupby('Region')['CurrentStock'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        
        # Utilization patterns
        df[f'UtilMA{window}'] = df.groupby('Region')['CurrentUtilization'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
    
    # Derived metrics
    df['StockToSales'] = df['CurrentStock'] / (df['SalesVolume'].rolling(7, min_periods=1).mean() + 1)
    df['TotalCost'] = df['LaborCost'] + df['MaterialCost'] + df['TransportCost']
    df['CostPerUnit'] = df['TotalCost'] / df['SalesVolume'].clip(lower=1)
    df['UtilizationEfficiency'] = df['SalesVolume'] / (df['CurrentUtilization'].clip(lower=0.1) * df['CurrentStock'])
    
    # Seasonal patterns
    df['SeasonalityIndex'] = df.groupby(['Month'])['SalesVolume'].transform('mean') / df['SalesVolume'].mean()
    
    # Region encoding
    df = pd.get_dummies(df, columns=['Region'], prefix='Region')
    
    return df

# 2. Analyze Features
def analyze_features(df):
    """Analyze and visualize important features"""
    # Correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation[['SalesVolume']].sort_values(by='SalesVolume', ascending=False),
                annot=True, cmap='coolwarm')
    plt.title('Feature Correlations with Sales Volume')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    plt.close()
    
    return correlation

# 3. Train Model
def train_model(df):
    """Train the prediction model"""
    # Select features
    features = [
        'DayOfWeek', 'Month', 'Quarter', 'IsWeekend',
        'SalesMA7', 'SalesMA30', 'SalesSTD7',
        'StockMA7', 'UtilMA7',
        'StockToSales', 'CostPerUnit',
        'UtilizationEfficiency', 'SeasonalityIndex'
    ] + [col for col in df.columns if col.startswith('Region_')]
    
    X = df[features]
    y = df['SalesVolume']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, feature_importance, train_score, test_score

# 4. Main execution
if __name__ == "__main__":
    # Load and augment data
    df = load_and_augment_data('automotive_supply_chain_data.csv')
    print("Data shape:", df.shape)
    
    # Analyze features
    correlation = analyze_features(df)
    print("\nTop correlations with SalesVolume:")
    print(correlation['SalesVolume'].sort_values(ascending=False).head(10))
    
    # Train model
    model, scaler, feature_importance, train_score, test_score = train_model(df)
    
    print("\nModel Performance:")
    print(f"Training Score: {train_score:.4f}")
    print(f"Testing Score: {test_score:.4f}")
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Save model and scaler
    joblib.dump(model, 'supply_chain_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    # Save sample data for reference
    df.to_csv('supply_chain_data.csv', index=False)
    print("Model, scaler, and sample data saved successfully!") 