import numpy as np

def inventory_calc(demand, lead_time):
    avg = np.mean(demand)
    std = np.std(demand)

    safety_stock = 1.65 * std * np.sqrt(lead_time)
    rop = avg * lead_time + safety_stock

    return round(safety_stock,2), round(rop,2)

def eoq_calc(avg_demand):
    ordering_cost = 50
    holding_cost = 2

    annual_demand = avg_demand * 365
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

    return round(eoq,2)