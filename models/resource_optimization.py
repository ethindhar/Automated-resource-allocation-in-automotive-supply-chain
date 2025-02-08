import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

class ResourceOptimizer:
    def __init__(self):
        self.resources = {}
        self.constraints = {}
        
    def optimize_allocation(self, demand_forecast, available_resources):
        """Optimize resource allocation based on demand forecast"""
        try:
            # Cost matrix calculation
            cost_matrix = self._calculate_cost_matrix(demand_forecast, available_resources)
            
            # Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            return {
                'optimal_allocation': self._format_allocation(row_ind, col_ind),
                'efficiency_score': self._calculate_efficiency(cost_matrix, row_ind, col_ind)
            }
        except Exception as e:
            raise Exception(f"Optimization error: {str(e)}") 