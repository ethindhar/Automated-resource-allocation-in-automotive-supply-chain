from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import sqlite3
import json
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from models.demand_forecasting import predict_demand
from models.optimization import optimize_resources
from models.supply_chain_model import train_and_predict
from models.supply_chain_predictor import SupplyChainPredictor
import joblib
from datetime import datetime, timedelta
from models.resource_optimization import ResourceOptimizer
from models.performance_metrics import SupplyChainMetrics
from scipy.optimize import linear_sum_assignment
import random  # For demo purposes. Replace with real data source.
import os

app = Flask(__name__)

# Load dataset and preprocess
df = pd.read_csv("automotive_supply_chain_data.csv")
df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime

# Database setup
def init_db():
    conn = sqlite3.connect('supply_chain.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS supply_chain_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            sales_volume REAL,
            current_stock REAL,
            current_utilization REAL,
            cost_per_unit REAL,
            region TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            predicted_sales REAL,
            actual_sales REAL,
            region TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

def get_db():
    conn = sqlite3.connect('supply_chain.db')
    conn.row_factory = sqlite3.Row
    return conn

# API to return optimized allocation
@app.route("/optimize", methods=["GET"])
def optimize_inventory():
    result = optimize_resources(df)
    return jsonify(result)

# API to return demand forecast
@app.route("/forecast", methods=["GET"])
def forecast_demand():
    try:
        forecast_result = predict_demand(df)
        return jsonify(forecast_result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'data': None
        }), 500

# Interactive Graphs
@app.route("/analytics")
def get_analytics():
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Get data from database
        cursor.execute("""
            SELECT * FROM supply_chain_data 
            ORDER BY date DESC 
            LIMIT 100
        """)
        rows = cursor.fetchall()
        
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=['id', 'date', 'sales_volume', 'current_stock', 
                                       'current_utilization', 'cost_per_unit', 'region', 'created_at'])
        
        analytics = {
            'inventory': {
                'stock_turnover': float(df['sales_volume'].sum() / df['current_stock'].mean()),
                'avg_stock': float(df['current_stock'].mean()),
                'stock_out_days': len(df[df['current_stock'] == 0])
            },
            'cost': {
                'total_cost': float(df['cost_per_unit'].sum()),
                'unit_cost': float(df['cost_per_unit'].mean()),
                'transport_cost': float(df['cost_per_unit'].sum() * 0.1)  # Example calculation
            },
            'utilization': {
                'avg_utilization': float(df['current_utilization'].mean()),
                'peak_utilization': float(df['current_utilization'].max()),
                'efficiency_rate': float(df['sales_volume'].mean() / df['current_stock'].mean())
            },
            'charts': {
                'stock_trend': {
                    'dates': df['date'].tolist(),
                    'stock': df['current_stock'].tolist(),
                    'sales': df['sales_volume'].tolist()
                },
                'regional_performance': df.groupby('region').agg({
                    'sales_volume': 'mean',
                    'current_utilization': 'mean',
                    'current_stock': 'mean'
                }).to_dict('index'),
                'cost_breakdown': {
                    'labels': ['Material', 'Labor', 'Transport'],
                    'values': [
                        float(df['cost_per_unit'].sum() * 0.6),
                        float(df['cost_per_unit'].sum() * 0.3),
                        float(df['cost_per_unit'].sum() * 0.1)
                    ]
                }
            }
        }
        
        conn.close()
        return jsonify({'status': 'success', 'data': analytics})
        
    except Exception as e:
        print(f"Error in get_analytics: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# Main UI
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/optimization")
def optimization_page():
    return render_template("optimization.html")

@app.route("/metrics")
def metrics_page():
    return render_template('metrics.html')

@app.route("/supply-chain-insights")
def supply_chain_insights():
    # 1. Inventory Analysis
    inventory_metrics = {
        'stock_turnover': df['SalesVolume'].sum() / df['CurrentStock'].mean(),
        'avg_stock_level': df['CurrentStock'].mean(),
        'stock_out_days': len(df[df['CurrentStock'] == 0]),
        'optimal_stock': df['SalesVolume'].mean() * 7  # 7-day buffer
    }
    
    # 2. Cost Analysis
    cost_analysis = {
        'total_costs': {
            'labor': df['LaborCost'].sum(),
            'material': df['MaterialCost'].sum(),
            'transport': df['TransportCost'].sum()
        },
        'cost_per_unit': (df['LaborCost'] + df['MaterialCost'] + df['TransportCost']).mean() / df['SalesVolume'].mean()
    }
    
    # 3. Utilization Analysis
    utilization_by_region = df.groupby('Region')['CurrentUtilization'].agg(['mean', 'min', 'max']).to_dict('index')
    
    # 4. Seasonal Patterns
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    seasonal_patterns = df.groupby('Month')['SalesVolume'].mean().to_dict()
    
    # 5. Create visualizations
    # Sales vs. Stock Level Trend
    stock_sales = px.line(df, x="Date", 
                         y=["CurrentStock", "SalesVolume"],
                         title="Stock Level vs. Sales Volume")
    
    # Cost Components Over Time
    costs = df.melt(id_vars=['Date'], 
                    value_vars=['LaborCost', 'MaterialCost', 'TransportCost'],
                    var_name='Cost Type', value_name='Amount')
    cost_trend = px.line(costs, x="Date", y="Amount", 
                        color="Cost Type",
                        title="Cost Components Over Time")
    
    # Utilization Heatmap by Region and Month
    util_heatmap = px.imshow(
        df.pivot_table(
            values='CurrentUtilization',
            index='Region',
            columns=pd.to_datetime(df['Date']).dt.month,
            aggfunc='mean'
        ),
        title="Utilization Heatmap by Region and Month"
    )
    
    # Convert plots to JSON
    graphs = {
        'stock_sales': json.dumps(stock_sales, cls=PlotlyJSONEncoder),
        'cost_trend': json.dumps(cost_trend, cls=PlotlyJSONEncoder),
        'util_heatmap': json.dumps(util_heatmap, cls=PlotlyJSONEncoder)
    }
    
    return render_template(
        "supply_chain_insights.html",
        inventory_metrics=inventory_metrics,
        cost_analysis=cost_analysis,
        utilization_by_region=utilization_by_region,
        seasonal_patterns=seasonal_patterns,
        graphs=graphs
    )

@app.route("/predictions")
def show_predictions():
    return render_template("predictions.html")

@app.route("/train-predict")
def train_predict():
    try:
        predictor = SupplyChainPredictor()
        scores = predictor.train_models(df)
        predictions = predictor.predict_future(df)
        
        return jsonify({
            'status': 'success',
            'data': {
                'model_scores': scores,
                'predictions': predictions
            },
            'message': 'Predictions generated successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'data': None
        })

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
            # Add region encoding
            'Region_Asia': 1 if input_values['region'] == 'Asia' else 0,
            'Region_Europe': 1 if input_values['region'] == 'Europe' else 0,
            'Region_North America': 1 if input_values['region'] == 'North America' else 0
        }
        
        pred_data.append(row)
    
    return pd.DataFrame(pred_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = {
            'date': request.form['date'],
            'stock': request.form['stock'],
            'utilization': request.form['utilization'],
            'sales': request.form['sales'],
            'cost': request.form['cost'],
            'region': request.form['region'],
            'days': int(request.form['days'])
        }
        
        response_data = predict_demand(input_values)
        return jsonify({
            'status': 'success',
            'data': response_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()
        
        # Extract input parameters
        resources = float(data.get('resources', 0))
        capacity = float(data.get('capacity', 0))
        labor = float(data.get('labor', 0))
        budget = float(data.get('budget', 0))

        # Calculate allocations based on inputs
        total_resources = resources + capacity + labor + budget
        production_alloc = (resources * 0.4 + capacity * 0.3) / total_resources * 100
        storage_alloc = (resources * 0.2 + capacity * 0.2) / total_resources * 100
        distribution_alloc = (resources * 0.2 + capacity * 0.3) / total_resources * 100
        labor_alloc = (labor * 0.8 + budget * 0.2) / total_resources * 100

        # Calculate efficiency scores
        resource_util = min(100, (resources / capacity) * 80)
        labor_eff = min(100, (labor / capacity) * 85)
        budget_eff = min(100, (budget / (resources + labor)) * 90)
        prod_balance = min(100, (capacity / (resources + labor)) * 88)
        
        # Calculate overall efficiency
        efficiency_score = (resource_util + labor_eff + budget_eff + prod_balance) / 4

        # Calculate KPIs
        inventory_turnover = (capacity * 1.2) / (resources * 0.8)
        labor_utilization = min(100, (labor / capacity) * 90)
        production_efficiency = min(100, (capacity / (resources + labor)) * 95)

        # Calculate component status
        base_inventory = resources * 0.5
        base_demand = capacity * 0.4
        component_status = {
            'Component A': {
                'inventory': int(base_inventory * 1.2),
                'demand': int(base_demand * 1.1),
                'production_rate': int(capacity * 0.3)
            },
            'Component B': {
                'inventory': int(base_inventory * 0.8),
                'demand': int(base_demand * 0.9),
                'production_rate': int(capacity * 0.4)
            },
            'Component C': {
                'inventory': int(base_inventory * 1.0),
                'demand': int(base_demand * 1.0),
                'production_rate': int(capacity * 0.3)
            }
        }

        # Calculate cost savings
        cost_savings = (efficiency_score / 100) * budget * 0.15

        # Generate forecast based on demand trends
        base_forecast = capacity * 0.8
        forecast = {
            'Week 1': int(base_forecast * 1.0),
            'Week 2': int(base_forecast * 1.1),
            'Week 3': int(base_forecast * 1.2),
            'Week 4': int(base_forecast * 1.15)
        }

        # Check for bottlenecks
        bottlenecks = []
        if resource_util < 60:
            bottlenecks.append("Resource utilization is below optimal levels")
        if labor_eff < 70:
            bottlenecks.append("Labor efficiency needs improvement")
        if budget_eff < 75:
            bottlenecks.append("Budget efficiency is suboptimal")

        result = {
            'status': 'success',
            'data': {
                'allocation': {
                    'Production': round(production_alloc, 1),
                    'Storage': round(storage_alloc, 1),
                    'Distribution': round(distribution_alloc, 1),
                    'Labor': round(labor_alloc, 1)
                },
                'efficiency_score': round(efficiency_score, 1),
                'efficiency_breakdown': {
                    'resource_utilization': round(resource_util, 1),
                    'labor_efficiency': round(labor_eff, 1),
                    'budget_efficiency': round(budget_eff, 1),
                    'production_balance': round(prod_balance, 1)
                },
                'cost_savings': round(cost_savings, 2),
                'component_status': component_status,
                'kpis': {
                    'inventory_turnover': round(inventory_turnover, 2),
                    'labor_utilization': round(labor_utilization, 1),
                    'production_efficiency': round(production_efficiency, 1)
                },
                'bottlenecks': bottlenecks,
                'forecast': forecast
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get-metrics', methods=['GET'])
def get_metrics():
    try:
        # Convert dates to datetime for time-based analysis
        df['Date'] = pd.to_datetime(df['Date'])
        last_180_days = df[df['Date'] >= (datetime.now() - timedelta(days=180))]
        
        # Calculate real utilization trends
        utilization_series = df.groupby('Date')['CurrentUtilization'].mean()
        
        # Calculate efficiency scores based on multiple factors
        labor_efficiency = (df['SalesVolume'] / (df['LaborCost'] + 1)).mean() * 85
        material_efficiency = (df['SalesVolume'] / (df['MaterialCost'] + 1)).mean() * 90
        transport_efficiency = (1 - df['TransportCost'] / df['MaterialCost'].max()) * 88
        stock_efficiency = (df['SalesVolume'] / (df['CurrentStock'] + 1)).mean() * 92
        
        # Calculate cost savings (comparing against baseline costs)
        baseline_cost = df['MaterialCost'].rolling(window=30).mean()
        actual_cost = df['MaterialCost']
        daily_savings = (baseline_cost - actual_cost).fillna(0)
        
        # Calculate performance metrics
        threshold = df['CurrentUtilization'].mean()
        efficient_days = len(df[df['CurrentUtilization'] >= threshold])
        improvement_days = len(df[df['CurrentUtilization'] < threshold])
        
        metrics_data = {
            'utilization': {
                'months': utilization_series.index.strftime('%Y-%m-%d').tolist(),
                'values': utilization_series.values.round(2).tolist()
            },
            'performance': {
                'efficient': efficient_days,
                'needs_improvement': improvement_days
            },
            'cost_savings': {
                'months': daily_savings.index.strftime('%Y-%m-%d').tolist(),
                'values': daily_savings.values.round(2).tolist()
            },
            'efficiency_metrics': {
                'categories': ['Labor Efficiency', 'Material Usage', 'Transport Efficiency', 'Stock Efficiency'],
                'values': [
                    min(100, round(labor_efficiency, 2)),
                    min(100, round(material_efficiency, 2)),
                    min(100, round(transport_efficiency, 2)),
                    min(100, round(stock_efficiency, 2))
                ]
            },
            'additional_insights': {
                'inventory_turnover': round(df['SalesVolume'].sum() / df['CurrentStock'].mean(), 2),
                'avg_fulfillment_rate': round((df['SalesVolume'] / df['CurrentStock']).mean() * 100, 2),
                'cost_reduction_trend': round((1 - df['MaterialCost'].tail(30).mean() / 
                                             df['MaterialCost'].head(30).mean()) * 100, 2),
                'peak_utilization_periods': df.groupby(df['Date'].dt.hour)['CurrentUtilization'].mean().nlargest(3).to_dict()
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': metrics_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/advanced-analytics', methods=['GET'])
def advanced_analytics():
    try:
        # Regional Performance Analysis
        regional_metrics = df.groupby('Region').agg({
            'SalesVolume': ['mean', 'max', 'std'],
            'CurrentStock': 'mean',
            'CurrentUtilization': 'mean',
            'MaterialCost': 'sum',
            'LaborCost': 'sum',
            'TransportCost': 'sum'
        }).round(2)

        # Time-based Analysis
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        
        # Seasonal Patterns
        seasonal_analysis = df.groupby('Month').agg({
            'SalesVolume': 'mean',
            'CurrentUtilization': 'mean',
            'MaterialCost': 'mean'
        }).round(2)

        # Supply Chain Health Score
        health_scores = {
            'inventory_health': min(100, (df['CurrentStock'] / df['SalesVolume']).mean() * 85),
            'cost_efficiency': min(100, (1 - (df['TransportCost'] / df['MaterialCost'])).mean() * 90),
            'operational_efficiency': min(100, df['CurrentUtilization'].mean()),
            'demand_satisfaction': min(100, (df['SalesVolume'] / df['CurrentStock']).mean() * 95)
        }

        # Risk Assessment
        risk_metrics = {
            'stock_out_risk': len(df[df['CurrentStock'] < df['SalesVolume'].mean()]) / len(df) * 100,
            'high_cost_periods': len(df[df['MaterialCost'] > df['MaterialCost'].mean()]) / len(df) * 100,
            'utilization_risk': len(df[df['CurrentUtilization'] > 90]) / len(df) * 100
        }

        # Trend Analysis
        rolling_metrics = {
            'sales_trend': df['SalesVolume'].rolling(window=7).mean().tail(30).tolist(),
            'cost_trend': df['MaterialCost'].rolling(window=7).mean().tail(30).tolist(),
            'utilization_trend': df['CurrentUtilization'].rolling(window=7).mean().tail(30).tolist()
        }

        # Performance Benchmarks
        benchmarks = {
            'top_performing_regions': df.groupby('Region')['SalesVolume'].mean().nlargest(3).to_dict(),
            'cost_efficient_regions': df.groupby('Region')['MaterialCost'].mean().nsmallest(3).to_dict(),
            'utilization_leaders': df.groupby('Region')['CurrentUtilization'].mean().nlargest(3).to_dict()
        }

        # Correlation Analysis
        correlation_matrix = df[['SalesVolume', 'CurrentStock', 'CurrentUtilization', 'MaterialCost']].corr().round(3)

        # Efficiency Metrics
        efficiency_metrics = {
            'resource_utilization': (df['SalesVolume'] / df['CurrentStock']).mean() * 100,
            'cost_per_unit': (df['MaterialCost'] + df['LaborCost'] + df['TransportCost']) / df['SalesVolume'],
            'operational_efficiency': df['SalesVolume'] / (df['CurrentUtilization'] * df['CurrentStock'])
        }

        return jsonify({
            'status': 'success',
            'data': {
                'regional_performance': regional_metrics.to_dict(),
                'seasonal_patterns': seasonal_analysis.to_dict(),
                'health_scores': health_scores,
                'risk_assessment': risk_metrics,
                'trends': rolling_metrics,
                'benchmarks': benchmarks,
                'correlation_analysis': correlation_matrix.to_dict(),
                'efficiency_metrics': {k: float(v.mean()) for k, v in efficiency_metrics.items()}
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route("/advanced-analytics-page")
def advanced_analytics_page():
    return render_template("advanced_analytics.html")

if __name__ == "__main__":
    app.run(debug=True)
