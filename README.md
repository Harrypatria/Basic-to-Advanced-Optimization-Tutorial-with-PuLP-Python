# 🧮 Mathematical Optimization with Python Tutorial

<div align="center">
  
![Python Optimization](https://raw.githubusercontent.com/coin-or/pulp/master/pulp/images/logo.png)

[![GitHub stars](https://img.shields.io/github/stars/yourusername/Python_Mathematical_Optimization?style=social)](https://github.com/yourusername/Python_Mathematical_Optimization/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/Python_Mathematical_Optimization?style=social)](https://github.com/yourusername/Python_Mathematical_Optimization/network/members)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 🌟 Overview

A comprehensive, production-ready mathematical optimization tutorial using Python libraries such as PuLP, SciPy, CVXPY, and Pyomo. This tutorial covers everything from basic linear programming to advanced optimization techniques for real-world problems in operations research, finance, logistics, and machine learning.

This repository contains real-world examples, proven modeling techniques, and industry best practices that you can immediately implement in your projects to solve complex optimization problems efficiently.

## ✨ Key Features

- **📚 Complete Learning Path**: Progressive examples from beginner to advanced optimization models
- **⚡ Multiple Solvers**: Interface with commercial and open-source solvers (CBC, GLPK, Gurobi, CPLEX)
- **🔄 Real-world Applications**: Supply chain optimization, portfolio management, resource allocation, scheduling
- **📈 Visualization**: Techniques for visualizing optimization models and solutions
- **🧪 Interactive Notebooks**: Run all examples directly in Google Colab
- **🏭 Industrial Applications**: Production planning, transportation problems, facility location
- **💹 Financial Optimization**: Portfolio optimization, risk management, option pricing
- **🛡️ Production-ready Code**: Industry best practices for implementing optimization models at scale

## 🚀 Quick Start

### Option 1: Run in Google Colab (No Installation Required)

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
  <a href="https://colab.research.google.com/drive/your-notebook-link" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Linear Programming In Colab" />
  </a>
  <a href="https://colab.research.google.com/drive/your-second-notebook-link" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Integer Programming In Colab" />
  </a>
  <a href="https://colab.research.google.com/drive/your-third-notebook-link" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Supply Chain Optimization In Colab" />
  </a>
  <a href="https://colab.research.google.com/drive/your-fourth-notebook-link" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Portfolio Optimization In Colab" />
  </a>
</p>

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Python_Mathematical_Optimization.git
cd Python_Mathematical_Optimization

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# This includes: pulp, scipy, cvxpy, pyomo, matplotlib, pandas, numpy

# Launch Jupyter notebook
jupyter notebook
```

## 📋 Notebooks Included

1. **Linear_Programming_Basics.ipynb** - Introduction to linear programming with PuLP
2. **Integer_Programming.ipynb** - Solving discrete optimization problems
3. **Mixed_Integer_Programming.ipynb** - Complex models with continuous and discrete variables
4. **Supply_Chain_Optimization.ipynb** - End-to-end supply chain modeling
5. **Portfolio_Optimization.ipynb** - Financial portfolio optimization
6. **Scheduling_Problems.ipynb** - Job shop and resource scheduling
7. **Transportation_Problems.ipynb** - Classical transportation and assignment problems
8. **Advanced_Techniques.ipynb** - Column generation, decomposition methods, and stochastic optimization

## 📋 Table of Contents

1. **Fundamentals**
   - Introduction to Mathematical Optimization
   - Linear Programming Formulation
   - PuLP Library Basics
   - Solving and Interpreting Results

2. **Intermediate Optimization**
   - Integer Programming
   - Mixed-Integer Linear Programming (MILP)
   - Binary Variables and Logical Constraints
   - Sensitivity Analysis

3. **Advanced Techniques**
   - Non-linear Optimization with CVXPY
   - Large-Scale Optimization Strategies
   - Multi-objective Optimization
   - Stochastic Programming
   - Robust Optimization

4. **Visualization and Analysis**
   - Visualizing Optimization Models
   - Sensitivity Analysis
   - Post-Optimization Analysis
   - Reporting and Dashboards

5. **Real-world Applications**
   - Supply Chain Network Design
   - Production Planning and Scheduling
   - Portfolio Optimization
   - Resource Allocation
   - Facility Location
   - Vehicle Routing Problems
   - Staff Scheduling and Rostering

6. **Production Best Practices**
   - Model Validation
   - Performance Tuning
   - Integration with Other Systems
   - Deployment Strategies
   - Maintenance and Updates

## 🔥 Optimization Models Showcase

| Model Type | Description | Applications |
|-----------|-------------|-----------------|
| Linear Programming | Optimize linear objective with linear constraints | Resource allocation, diet problems, blending |
| Integer Programming | Problems with discrete variables | Scheduling, assignment, knapsack problems |
| Mixed-Integer Programming | Combination of continuous and discrete variables | Facility location, supply chain design |
| Network Flow | Optimization on network structures | Transportation, logistics, communications |
| Multi-objective | Optimizing several conflicting objectives | Portfolio optimization, engineering design |
| Non-linear Programming | Non-linear objective or constraints | Engineering design, ML hyperparameter tuning |
| Stochastic Programming | Optimization under uncertainty | Financial planning, energy systems |
| Robust Optimization | Solutions resistant to parameter uncertainty | Supply chain under disruption risk |

## 💡 Real-world Example: Production Planning

```python
import pulp as pl

# Create a linear programming problem
model = pl.LpProblem("Production_Planning", pl.LpMaximize)

# Products
products = ["Product_A", "Product_B", "Product_C"]
profit = {"Product_A": 10, "Product_B": 12, "Product_C": 14}  # profit per unit
demand = {"Product_A": 100, "Product_B": 80, "Product_C": 50}  # max demand

# Resources
resources = ["Labor", "Material_X", "Material_Y"]
available = {"Labor": 480, "Material_X": 320, "Material_Y": 250}  # available hours/units

# Resource usage per unit of product
usage = {
    ("Product_A", "Labor"): 5,
    ("Product_B", "Labor"): 4,
    ("Product_C", "Labor"): 6,
    ("Product_A", "Material_X"): 3,
    ("Product_B", "Material_X"): 5,
    ("Product_C", "Material_X"): 2,
    ("Product_A", "Material_Y"): 2,
    ("Product_B", "Material_Y"): 1,
    ("Product_C", "Material_Y"): 4
}

# Decision variables - how many of each product to make
production = {p: pl.LpVariable(f"Produce_{p}", lowBound=0, cat='Integer') for p in products}

# Objective function: maximize profit
model += pl.lpSum([profit[p] * production[p] for p in products]), "Total_Profit"

# Resource constraints
for r in resources:
    model += pl.lpSum([usage[(p, r)] * production[p] for p in products]) <= available[r], f"Total_{r}"

# Demand constraints
for p in products:
    model += production[p] <= demand[p], f"Demand_{p}"

# Solve the model
model.solve()

# Print the results
print(f"Status: {pl.LpStatus[model.status]}")
print("\nOptimal Production Plan:")
for p in products:
    print(f"{p}: {production[p].value()} units")

print(f"\nTotal Profit: ${pl.value(model.objective)}")

# Resource usage
print("\nResource Usage:")
for r in resources:
    total_used = sum(usage[(p, r)] * production[p].value() for p in products)
    print(f"{r}: {total_used} / {available[r]} ({total_used/available[r]*100:.1f}%)")
```

## 🔧 Advanced Use Cases

### Supply Chain Network Optimization

```python
import pulp as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Create model
model = pl.LpProblem("Supply_Chain_Optimization", pl.LpMinimize)

# Sets
factories = ["Factory_A", "Factory_B", "Factory_C"]
warehouses = ["Warehouse_1", "Warehouse_2", "Warehouse_3", "Warehouse_4"]
customers = ["Customer_X", "Customer_Y", "Customer_Z", "Customer_W"]

# Parameters
production_capacity = {"Factory_A": 500, "Factory_B": 700, "Factory_C": 600}
warehouse_capacity = {"Warehouse_1": 400, "Warehouse_2": 500, "Warehouse_3": 300, "Warehouse_4": 600}
customer_demand = {"Customer_X": 300, "Customer_Y": 400, "Customer_Z": 350, "Customer_W": 250}

# Costs
production_cost = {"Factory_A": 5, "Factory_B": 6, "Factory_C": 4}
factory_to_warehouse_cost = {
    ("Factory_A", "Warehouse_1"): 3, ("Factory_A", "Warehouse_2"): 2, 
    ("Factory_A", "Warehouse_3"): 5, ("Factory_A", "Warehouse_4"): 4,
    ("Factory_B", "Warehouse_1"): 2, ("Factory_B", "Warehouse_2"): 3, 
    ("Factory_B", "Warehouse_3"): 2, ("Factory_B", "Warehouse_4"): 3,
    ("Factory_C", "Warehouse_1"): 4, ("Factory_C", "Warehouse_2"): 3, 
    ("Factory_C", "Warehouse_3"): 2, ("Factory_C", "Warehouse_4"): 2
}
warehouse_to_customer_cost = {
    ("Warehouse_1", "Customer_X"): 2, ("Warehouse_1", "Customer_Y"): 3, 
    ("Warehouse_1", "Customer_Z"): 4, ("Warehouse_1", "Customer_W"): 5,
    ("Warehouse_2", "Customer_X"): 3, ("Warehouse_2", "Customer_Y"): 2, 
    ("Warehouse_2", "Customer_Z"): 3, ("Warehouse_2", "Customer_W"): 4,
    ("Warehouse_3", "Customer_X"): 5, ("Warehouse_3", "Customer_Y"): 3, 
    ("Warehouse_3", "Customer_Z"): 2, ("Warehouse_3", "Customer_W"): 3,
    ("Warehouse_4", "Customer_X"): 4, ("Warehouse_4", "Customer_Y"): 4, 
    ("Warehouse_4", "Customer_Z"): 3, ("Warehouse_4", "Customer_W"): 2
}

# Decision variables
production = {f: pl.LpVariable(f"Produce_{f}", lowBound=0, cat='Continuous') for f in factories}
ship_to_warehouse = {(f, w): pl.LpVariable(f"Ship_{f}_to_{w}", lowBound=0, cat='Continuous') 
                    for f in factories for w in warehouses}
ship_to_customer = {(w, c): pl.LpVariable(f"Ship_{w}_to_{c}", lowBound=0, cat='Continuous') 
                   for w in warehouses for c in customers}

# Objective function: minimize total cost
model += (
    pl.lpSum([production_cost[f] * production[f] for f in factories]) +
    pl.lpSum([factory_to_warehouse_cost[(f, w)] * ship_to_warehouse[(f, w)] 
             for f in factories for w in warehouses]) +
    pl.lpSum([warehouse_to_customer_cost[(w, c)] * ship_to_customer[(w, c)] 
             for w in warehouses for c in customers])
)

# Constraints
# Factory capacity
for f in factories:
    model += production[f] <= production_capacity[f], f"Capacity_{f}"

# Production balance
for f in factories:
    model += production[f] == pl.lpSum([ship_to_warehouse[(f, w)] for w in warehouses]), f"Balance_{f}"

# Warehouse capacity
for w in warehouses:
    model += pl.lpSum([ship_to_warehouse[(f, w)] for f in factories]) <= warehouse_capacity[w], f"Capacity_{w}"

# Warehouse flow balance
for w in warehouses:
    model += pl.lpSum([ship_to_warehouse[(f, w)] for f in factories]) == \
            pl.lpSum([ship_to_customer[(w, c)] for c in customers]), f"Flow_{w}"

# Customer demand
for c in customers:
    model += pl.lpSum([ship_to_customer[(w, c)] for w in warehouses]) == customer_demand[c], f"Demand_{c}"

# Solve the model
model.solve()

print(f"Status: {pl.LpStatus[model.status]}")
print(f"Total Cost: ${pl.value(model.objective)}")

# Visualize the network flow
def plot_network_flow():
    G = nx.DiGraph()
    
    # Add nodes
    for f in factories:
        G.add_node(f, node_type='factory')
    for w in warehouses:
        G.add_node(w, node_type='warehouse')
    for c in customers:
        G.add_node(c, node_type='customer')
    
    # Add edges with flow values
    for f in factories:
        for w in warehouses:
            flow = ship_to_warehouse[(f, w)].value()
            if flow > 0:
                G.add_edge(f, w, weight=flow)
    
    for w in warehouses:
        for c in customers:
            flow = ship_to_customer[(w, c)].value()
            if flow > 0:
                G.add_edge(w, c, weight=flow)
    
    # Plot the graph
    plt.figure(figsize=(12, 8))
    pos = {**{f: (0, i) for i, f in enumerate(factories)},
           **{w: (1, i) for i, w in enumerate(warehouses)},
           **{c: (2, i) for i, c in enumerate(customers)}}
    
    node_colors = {**{f: 'lightblue' for f in factories},
                  **{w: 'lightgreen' for w in warehouses},
                  **{c: 'salmon' for c in customers}}
    
    nx.draw_networkx_nodes(G, pos, node_color=[node_colors[n] for n in G.nodes])
    nx.draw_networkx_labels(G, pos)
    
    edge_widths = [G[u][v]['weight']/50 for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray')
    
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Supply Chain Network Flow")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_network_flow()
```

### Portfolio Optimization (Markowitz Model)

```python
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf

# Load historical stock data (example)
# This would normally come from a financial data provider
np.random.seed(42)
n_assets = 10
n_days = 1000
returns = np.random.randn(n_days, n_assets) * 0.01  # Daily returns
returns = returns + np.linspace(0.0005, 0.001, n_assets)  # Add different means

# Calculate expected returns and covariance
mu = np.mean(returns, axis=0)

# Use Ledoit-Wolf shrinkage estimator for robust covariance estimation
lw = LedoitWolf().fit(returns)
Sigma = lw.covariance_

# Portfolio optimization model
w = cp.Variable(n_assets)
gamma = cp.Parameter(nonneg=True)  # Risk aversion parameter

# Expected return
ret = mu.T @ w

# Risk (variance)
risk = cp.quad_form(w, Sigma)

# Objective: maximize return - gamma*risk (utility function)
objective = cp.Maximize(ret - gamma * risk)

# Constraints
constraints = [
    cp.sum(w) == 1,              # Fully invested
    w >= 0.01,                   # Minimum position size
    w <= 0.25                    # Maximum position size
]

# Define the problem
problem = cp.Problem(objective, constraints)

# Calculate the efficient frontier
gamma_vals = np.logspace(-1, 2, 100)
returns_ef = []
risks_ef = []
weights_ef = []

for g in gamma_vals:
    gamma.value = g
    problem.solve()
    
    if problem.status == 'optimal':
        returns_ef.append(ret.value)
        risks_ef.append(np.sqrt(risk.value))  # Convert variance to std dev
        weights_ef.append(w.value)

# Plot efficient frontier
plt.figure(figsize=(12, 8))
plt.plot(risks_ef, returns_ef, 'b-')
plt.scatter(risks_ef, returns_ef, c=gamma_vals, cmap='viridis', s=30)
plt.colorbar(label='Risk Aversion (gamma)')

# Individual assets
asset_risks = np.sqrt(np.diag(Sigma))
plt.scatter(asset_risks, mu, c='red', s=120, marker='*')

# Minimum variance portfolio
min_var_idx = np.argmin(risks_ef)
plt.scatter(risks_ef[min_var_idx], returns_ef[min_var_idx], c='green', s=160, marker='o')

# Maximum Sharpe ratio portfolio (assuming Rf=0)
sharpe_ratios = np.array(returns_ef) / np.array(risks_ef)
max_sharpe_idx = np.argmax(sharpe_ratios)
plt.scatter(risks_ef[max_sharpe_idx], returns_ef[max_sharpe_idx], c='orange', s=160, marker='D')

plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Plot asset allocation for selected portfolios
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Minimum variance portfolio
asset_names = [f'Asset {i+1}' for i in range(n_assets)]
axs[0].pie(weights_ef[min_var_idx], labels=asset_names, autopct='%1.1f%%')
axs[0].set_title('Minimum Variance Portfolio')

# Maximum Sharpe ratio portfolio
axs[1].pie(weights_ef[max_sharpe_idx], labels=asset_names, autopct='%1.1f%%')
axs[1].set_title('Maximum Sharpe Ratio Portfolio')

plt.tight_layout()
plt.show()

print(f"Minimum Variance Portfolio - Return: {returns_ef[min_var_idx]:.4f}, Risk: {risks_ef[min_var_idx]:.4f}")
print(f"Maximum Sharpe Ratio Portfolio - Return: {returns_ef[max_sharpe_idx]:.4f}, Risk: {risks_ef[max_sharpe_idx]:.4f}, Sharpe: {sharpe_ratios[max_sharpe_idx]:.4f}")
```

### Employee Scheduling Problem

```python
import pulp as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a new optimization model
model = pl.LpProblem("Employee_Scheduling", pl.LpMinimize)

# Parameters
days = list(range(7))  # 0=Monday, 6=Sunday
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
shifts = ["Morning", "Afternoon", "Night"]
employees = ["Employee" + str(i+1) for i in range(10)]

# Shift requirements - Number of employees needed per shift per day
requirements = {
    (0, "Morning"): 3, (0, "Afternoon"): 2, (0, "Night"): 1,
    (1, "Morning"): 3, (1, "Afternoon"): 2, (1, "Night"): 1,
    (2, "Morning"): 3, (2, "Afternoon"): 2, (2, "Night"): 1,
    (3, "Morning"): 3, (3, "Afternoon"): 2, (3, "Night"): 1,
    (4, "Morning"): 3, (4, "Afternoon"): 2, (4, "Night"): 1,
    (5, "Morning"): 2, (5, "Afternoon"): 2, (5, "Night"): 2,
    (6, "Morning"): 2, (6, "Afternoon"): 2, (6, "Night"): 2
}

# Shift preferences (higher number = more preferred)
preferences = {e: {} for e in employees}

# Generate some preferences (in a real scenario, these would come from employees)
import random
random.seed(42)
for e in employees:
    for d in days:
        for s in shifts:
            # Weekend night shifts are less desirable
            if d >= 5 and s == "Night":
                preferences[e][(d, s)] = random.randint(1, 3)
            # Weekday preferences
            else:
                preferences[e][(d, s)] = random.randint(1, 10)

# Cost of assigning an employee contrary to preference (10 - preference)
cost = {(e, d, s): 10 - preferences[e][(d, s)] for e in employees for d in days for s in shifts}

# Decision Variables - 1 if employee e is assigned to shift s on day d
x = {(e, d, s): pl.LpVariable(f"Assign_{e}_Day{d}_{s}", cat='Binary') 
     for e in employees for d in days for s in shifts}

# Objective Function - Minimize cost (maximize preferences)
model += pl.lpSum([cost[(e, d, s)] * x[(e, d, s)] for e in employees for d in days for s in shifts])

# Constraint 1: Meet shift requirements
for d in days:
    for s in shifts:
        model += pl.lpSum([x[(e, d, s)] for e in employees]) >= requirements[(d, s)], f"Requirement_Day{d}_{s}"

# Constraint 2: Each employee works at most 1 shift per day
for e in employees:
    for d in days:
        model += pl.lpSum([x[(e, d, s)] for s in shifts]) <= 1, f"OneShift_{e}_Day{d}"

# Constraint 3: Each employee works between 3 and 5 shifts per week
for e in employees:
    model += pl.lpSum([x[(e, d, s)] for d in days for s in shifts]) >= 3, f"MinShifts_{e}"
    model += pl.lpSum([x[(e, d, s)] for d in days for s in shifts]) <= 5, f"MaxShifts_{e}"

# Constraint 4: No more than 3 consecutive days of work
for e in employees:
    for d in range(5):  # Can only start a 3-consecutive-day period up to day 4
        model += (
            pl.lpSum([x[(e, d+i, s)] for i in range(4) for s in shifts]) <= 3,
            f"ConsecutiveDays_{e}_From{d}"
        )

# Constraint 5: If working a night shift, no morning shift the next day
for e in employees:
    for d in range(6):  # All days except the last
        model += (
            x[(e, d, "Night")] + x[(e, (d+1), "Morning")] <= 1,
            f"NoMorningAfterNight_{e}_Day{d}"
        )

# Constraint 6: At least one weekend day off every two weeks
# (For simplicity, we're just ensuring one weekend day off in this one-week schedule)
for e in employees:
    model += (
        x[(e, 5, "Morning")] + x[(e, 5, "Afternoon")] + x[(e, 5, "Night")] +
        x[(e, 6, "Morning")] + x[(e, 6, "Afternoon")] + x[(e, 6, "Night")] <= 1,
        f"WeekendOff_{e}"
    )

# Solve the model
model.solve()

print(f"Status: {pl.LpStatus[model.status]}")
print(f"Total Cost (Lower = Better): {pl.value(model.objective)}")

# Create a schedule dataframe for visualization
schedule = pd.DataFrame(index=employees, columns=[(d, s) for d in days for s in shifts])
for e in employees:
    for d in days:
        for s in shifts:
            schedule.loc[e, (d, s)] = 1 if x[(e, d, s)].value() > 0.5 else 0

# Reshape for better visualization
schedule_viz = pd.DataFrame(columns=['Employee', 'Day', 'Shift', 'Working'])
for e in employees:
    for d in days:
        for s in shifts:
            schedule_viz = pd.concat([schedule_viz, pd.DataFrame({
                'Employee': [e], 
                'Day': [day_names[d]], 
                'Shift': [s], 
                'Working': [1 if x[(e, d, s)].value() > 0.5 else 0]
            })], ignore_index=True)

# Pivot for heatmap
schedule_pivot = schedule_viz.pivot_table(
    index='Employee', 
    columns=['Day', 'Shift'], 
    values='Working', 
    aggfunc='sum'
)

# Plot
plt.figure(figsize=(15, 8))
sns.heatmap(schedule_pivot, cmap='YlGnBu', cbar=False, linewidths=.5)
plt.title('Weekly Employee Schedule')
plt.tight_layout()
plt.show()

# Print schedule
print("\nWeekly Schedule:")
for d in days:
    print(f"\n{day_names[d]}:")
    for s in shifts:
        working = [e for e in employees if x[(e, d, s)].value() > 0.5]
        print(f"  {s}: {', '.join(working)}")
```

## 📚 Learning Path

This tutorial is designed to progressively build your mathematical optimization expertise:

1. **Day 1**: Linear programming fundamentals and PuLP basics
2. **Day 2**: Integer programming and binary variables
3. **Day 3**: Supply chain and logistics optimization
4. **Day 4**: Multi-objective optimization
5. **Day 5**: Financial portfolio optimization
6. **Day 6**: Advanced scheduling problems
7. **Day 7**: Large-scale optimization techniques

## 🤔 Why Python for Mathematical Optimization?

- **Accessibility**: High-level Python APIs make mathematical optimization accessible to non-experts
- **Flexibility**: Choose from multiple solvers (open-source and commercial)
- **Integration**: Seamlessly integrates with data science and analytics workflows
- **Visualization**: Powerful visualization tools for understanding optimization results
- **Open-Source Ecosystem**: Robust libraries with strong community support
- **Industry Support**: Used in production systems across industries worldwide
- **Learning Curve**: Much easier to learn than specialized modeling languages

## 📊 Industry Applications

- **Manufacturing**: Production planning, cutting stock problems, assembly line balancing
- **Logistics**: Vehicle routing, warehouse location, inventory management
- **Finance**: Portfolio optimization, risk management, option pricing
- **Healthcare**: Staff scheduling, patient assignment, resource allocation
- **Energy**: Generation planning, grid optimization, renewable energy integration
- **Retail**: Pricing optimization, shelf space allocation, assortment planning
- **Telecommunications**: Network design, cell tower placement, frequency assignment

## 🛠️ Troubleshooting Common Issues

<details>
<summary><b>Model Infeasibility</b></summary>

When your model returns as "Infeasible":

```python
# Solution 1: Implement an elastic model to find constraints causing infeasibility
def create_elastic_model(original_model):
    # Create a copy of the original model
    elastic_model = pl.LpProblem("Elastic_" + original_model.name, pl.LpMinimize)
    
    # Add all variables from the original model
    for v in original_model.variables():
        elastic_model.addVariable(v)
    
    # Add elastic variables for each constraint
    elastic_vars = {}
    for i, constraint in enumerate(original_model.constraints.values()):
        # Create positive and negative elastic variables
        elastic_vars[i] = (
            pl.LpVariable(f"Elastic_plus_{i}", lowBound=0),
            pl.LpVariable(f"Elastic_minus_{i}", lowBound=0)
        )
    
    # Add modified constraints with elastic variables
    for i, (name, constraint) in enumerate(original_model.constraints.items()):
        if constraint.sense == 1:  # <=
            elastic_model.addConstraint(
                constraint.e + elastic_vars[i][0] - elastic_vars[i][1] <= constraint.constant,
                name=name
            )
        elif constraint.sense == -1:  # >=
            elastic_model.addConstraint(
                constraint.e + elastic_vars[i][0] - elastic_vars[i][1] >= constraint.constant,
                name=name
            )
        else:  # ==
            elastic_model.addConstraint(
                constraint.e + elastic_vars[i][0] - elastic_vars[i][1] == constraint.constant,
                name=name
            )
    
    # Set objective to minimize sum of elastic variables (weighted)
    elastic_model += pl.lpSum([ev[0] + ev[1] for ev in elastic_vars.values()])
    
    return elastic_model, elastic_vars

# Solution 2: Add constraint relaxation systematically
def find_infeasible_constraint_set(model):
    """Find a minimal set of constraints causing infeasibility"""
    constraint_names = list(model.constraints.keys())
    
    # First check if removing any single constraint makes the model feasible
    for name in constraint_names:
        test_model = model.copy()
        del test_model.constraints[name]
        test_model.solve()
        if test_model.status == 1:  # Optimal
            print(f"Removing constraint {name} makes the model feasible")
            return [name]
    
    # If no single constraint is the cause, try pairs
    from itertools import combinations
    for pair in combinations(constraint_names, 2):
        test_model = model.copy()
        del test_model.constraints[pair[0]]
        del test_model.constraints[pair[1]]
        test_model.solve()
        if test_model.status == 1:  # Optimal
            print(f"Removing constraints {pair} makes the model feasible")
            return list(pair)
    
    return "Complex infeasibility requiring further analysis"

# Solution 3: Check for conflicting bounds or constraints
def check_for_contradictions(model):
    # Check variable bounds for contradictions
    for v in model.variables():
        if v.lowBound is not None and v.upBound is not None:
            if v.lowBound > v.upBound:
                print(f"Contradiction found: {v.name} has lower bound {v.lowBound} > upper bound {v.upBound}")
    
    # Look for obviously contradictory constraints (simplified example)
    constraints = list(model.constraints.values())
    variables = model.variables()
    
    for i, c1 in enumerate(constraints):
        for c2 in constraints[i+1:]:
            # This is a simplified check - in practice, this would be more sophisticated
            if str(c1.e) == str(c2.e) and c1.constant != c2.constant:
                if c1.sense == 0 and c2.sense == 0:  # Both equality constraints
                    print(f"Potential contradiction: Same expression with different constants")
                    print(f"  {c1.e} == {c1.constant}")
                    print(f"  {c2.e} == {c2.constant}")
```
