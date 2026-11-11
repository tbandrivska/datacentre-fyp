from scipy.optimize import linprog
import pulp

# Sets and time
gateways = ["GW_East", "GW_Central", "GW_West"]
datacenters = ["DC_Virginia", "DC_Texas", "DC_Arizona", "DC_Washington"]
modes = [1, 2, 3]  # 1 Free Air, 2 Hybrid, 3 Mechanical 
hours = list(range(24))

# Workload params
#SCENARIO_SPECIFIC!!!!!!!
# Token geometry per inference request
T_pre = 1000  # Input/Prefill tokens per request
T_dec = 200   # Output/Decode tokens per request

# Energy coefficients (Joules per token)
# Reflecting Kazybekov (2026) empirical ~8x decode-penalty ratio
E_pre = 0.05  # J/token
E_dec = 0.40  # J/token

# Convert per-request token energy from Joules to kWh
# (1 kWh = 3,600,000 Joules)
kWh_per_request = ((T_pre * E_pre) + (T_dec * E_dec)) / 3.6e6

# IT Power Capacity per Data Center (kW)
capacity_IT_kW = {
    "DC_Virginia":   5000.0,
    "DC_Texas":      5000.0,
    "DC_Arizona":    4000.0,
    "DC_Washington": 4000.0
}
#PUE cooling mode params
pue_mode = {1: 1.05, 2: 1.15, 3: 1.35}
# Latency params
# Maximum allowable round-trip network propagation delay (ms)
SLO_max = 80.0

# Network Latency Matrix (ms) from Gateways to Data Centers
latency_ms = {
    "GW_East":    {"DC_Virginia": 15, "DC_Texas": 45, "DC_Arizona": 75, "DC_Washington": 90},
    "GW_Central": {"DC_Virginia": 40, "DC_Texas": 20, "DC_Arizona": 35, "DC_Washington": 65},
    "GW_West":    {"DC_Virginia": 85, "DC_Texas": 55, "DC_Arizona": 25, "DC_Washington": 30}
}

# Demand, weather
# Diurnal request arrival pattern (spikes during local day, drops at night)
diurnal_multiplier = [
    0.4, 0.3, 0.2, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.2,
    1.1, 1.1, 1.0, 1.0, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4
]

# Total incoming request volume (requests per hour per gateway)
demand_requests = {
    j: [int(120000 * mult) for mult in diurnal_multiplier]
    for j in gateways
}

# Day-Ahead Electricity Tariffs ($/kWh) per hour
tariff_usd_per_kWh = {
    "DC_Virginia":   [0.08 if 8 <= t <= 20 else 0.05 for t in hours],
    "DC_Texas":      [0.09 if 12 <= t <= 18 else 0.04 for t in hours],
    "DC_Arizona":    [0.10 if 13 <= t <= 19 else 0.045 for t in hours],
    "DC_Washington": [0.03 for _ in hours]  # Constant cheap hydro tariff
}

# Hourly wet-bulb temperature (°C) & Stage 1 linear PUE model
temp_wb = {
    "DC_Virginia":   [15 + 4 * (t / 24) for t in hours],
    "DC_Texas":      [22 + 7 * (t / 24) for t in hours],
    "DC_Arizona":    [24 + 9 * (t / 24) for t in hours],
    "DC_Washington": [8 + 3 * (t / 24) for t in hours]
}
#Direct WUE for mode 3 (mechanical cooling) as a function of wet-bulb temperature in celcius
def get_wue_dir_mode3(t_wb):
    return max(0.0, -0.0006143 * (t_wb**2) + 0.03386 * t_wb + 1.24)

pue_model = {
    i: [1.05 + 0.005 * temp_wb[i][t] for t in hours]
    for i in datacenters
}


#  MODEL FORMULATION (PuLP)
model = pulp.LpProblem("Stage1_SpatioTemporal_GLB", pulp.LpMinimize)
# Decision Variable: x[j, i, t] = requests routed from Gateway j to Data Center i at hour t
x = {
    (j, i, t): pulp.LpVariable(
        f"Workload_Route_{j}_{i}_{t}",
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    for j in gateways
    for i in datacenters
    for t in hours
}
# Objective Function: Minimise total system operational electricity cost ($)
# Total Power (kW) = IT Power * PUE
# Total Cost ($) = sum_i sum_t ( Tariff_i,t * Total_Power_i,t )
model += pulp.lpSum(
    tariff_usd_per_kWh[i][t] * 
    (pulp.lpSum(x[j, i, t] for j in gateways) * kWh_per_request) * 
    pue_model[i][t]
    for i in datacenters for t in hours
), "Total_Operational_Electricity_Cost"


# CONSTRAINTS
# Constraint 1: Demand Satisfaction (All gateway queries must be assigned)
for j in gateways:
    for t in hours:
        model += (
            pulp.lpSum(x[j, i, t] for i in datacenters) == demand_requests[j][t],
            f"Demand_Satisfaction_{j}_t{t}"
        )

# Constraint 2: Latency Reachability Filter (Enforce SLO limits)
for j in gateways:
    for i in datacenters:
        if latency_ms[j][i] > SLO_max:
            for t in hours:
                model += (
                    x[j, i, t] == 0,
                    f"SLO_Reachability_Filter_{j}_{i}_t{t}"
                )

# Constraint 3: IT Hardware Power Capacity Limit (kW)
for i in datacenters:
    for t in hours:
        model += (
            pulp.lpSum(x[j, i, t] for j in gateways) * kWh_per_request <= capacity_IT_kW[i],
            f"IT_Capacity_Limit_{i}_t{t}"
        )

# solving
status = model.solve(pulp.PULP_CBC_CMD(msg=False))

print(f"Solver Status: {pulp.LpStatus[status]}")
print(f"Total Operational Electricity Cost: ${pulp.value(model.objective):,.2f}\n")

# Aggregated summary per data center
print(f"{'Data Center':<15} | {'Routed Requests':<15} | {'Total IT Energy (kWh)':<22} | {'Total Facility Cost ($)':<22}")
print("-" * 80)

for i in datacenters:
    total_requests_i = sum(x[j, i, t].varValue for j in gateways for t in hours)
    total_it_kwh_i = total_requests_i * kWh_per_request
    total_cost_i = sum(
        tariff_usd_per_kWh[i][t] * 
        (sum(x[j, i, t].varValue for j in gateways) * kWh_per_request) * 
        pue_model[i][t]
        for t in hours
    )
    print(f"{i:<15} | {total_requests_i:<15,.0f} | {total_it_kwh_i:<22,.2f} | ${total_cost_i:<22,.2f}")