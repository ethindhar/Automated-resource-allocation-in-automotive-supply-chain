from scipy.optimize import linprog

def optimize_resources(df):
    # Define cost function (Material + Transport)
    cost = df["MaterialCost"] + df["TransportCost"]

    # Constraints: Stock should not be less than Reorder Level
    constraints = df["CurrentStock"] - df["ReorderLevel"]

    # Solve linear programming problem
    result = linprog(cost, A_eq=[constraints], b_eq=[0], method="highs")

    df["OptimizedAllocation"] = result.x if result.success else "No feasible solution"
    
    return df[["CurrentStock", "ReorderLevel", "OptimizedAllocation"]].to_dict(orient="records")
