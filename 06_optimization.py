"""
Week 4 Task - Optimization Strategy
---------------------------------------
Uses the predictive-model insight (delay is driven heavily by distance,
weekend dispatch, and shipment weight — NOT primarily by mode choice, per
the feature-importance chart) to formulate a mode-allocation optimization
problem:

    Given a lane's monthly shipment volume, decide how many shipments to
    route via each transport mode (Road, Rail, Air, Sea) to MINIMIZE total
    transportation cost, subject to:
        - all shipments for the lane must be allocated
        - each mode has a maximum monthly handling capacity
        - the volume-weighted average predicted delay must stay under
          a target SLA (e.g. 6 hours)

This is a linear program (LP), solved with scipy.optimize.linprog.
We compare the CURRENT mode mix (observed in the dataset) against the
OPTIMIZED mix for the busiest lane in the dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linprog

PALETTE = ["#2E5EAA", "#E8871E", "#4CA64C", "#C0392B"]
BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "model_charts"
OUT.mkdir(exist_ok=True)

raw_data_path = BASE_DIR / "logistics_dataset.csv"
if raw_data_path.exists():
    df = pd.read_csv(raw_data_path, parse_dates=["shipment_date"])
else:
    engineered_data_path = BASE_DIR / "model_dataset.csv"
    if not engineered_data_path.exists():
        raise FileNotFoundError(
            f"No input dataset found. Expected {raw_data_path.name} or "
            f"{engineered_data_path.name} in {BASE_DIR}."
        )

    # The raw logistics file is optional in this workspace. Build the minimum
    # planning fields from the engineered model dataset when it is unavailable.
    df = pd.read_csv(engineered_data_path)
    df["transport_mode"] = "Air"
    for mode, column in {
        "Road": "transport_mode_Road",
        "Rail": "transport_mode_Rail",
        "Sea": "transport_mode_Sea",
    }.items():
        if column in df.columns:
            df.loc[df[column].astype(bool), "transport_mode"] = mode

    estimated_cost_per_km = {"Road": 8.0, "Rail": 4.0, "Air": 20.0, "Sea": 2.0}
    df["transportation_cost_inr"] = (
        df["distance_km"] * df["transport_mode"].map(estimated_cost_per_km)
    )
    print("Warning: logistics_dataset.csv not found; using estimated mode costs.")

# ---------------------------------------------------------------
# 1. Pick the busiest lane (origin -> destination pair) as a case study
# ---------------------------------------------------------------
if {"origin_city", "destination_city"}.issubset(df.columns):
    df["lane"] = df["origin_city"] + " -> " + df["destination_city"]
else:
    df["lane"] = "All shipments"
busiest_lane = df["lane"].value_counts().idxmax()
lane_df = df[df["lane"] == busiest_lane]
print(f"Case-study lane: {busiest_lane}  ({len(lane_df)} historical shipments)")

# ---------------------------------------------------------------
# 2. Per-mode cost & delay benchmarks (from historical + model-informed data)
#    avg_cost_per_shipment: observed average cost per shipment by mode
#    avg_delay: observed average delay by mode (model confirms mode itself
#    is a minor driver vs. distance/weight/weekend, so these are used as
#    planning benchmarks for this lane specifically)
# ---------------------------------------------------------------
mode_stats = df.groupby("transport_mode").agg(
    avg_cost=("transportation_cost_inr", "mean"),
    avg_delay=("delay_hrs", "mean"),
).reindex(["Road", "Rail", "Air", "Sea"])
print("\nMode benchmarks (all lanes):")
print(mode_stats.round(2).to_string())

modes = list(mode_stats.index)
cost = mode_stats["avg_cost"].values
delay = mode_stats["avg_delay"].values

# Monthly shipment volume for this lane (scaled up for a realistic monthly
# planning volume; the historical sample is a full-year snapshot)
TOTAL_SHIPMENTS = 40          # monthly volume to allocate across modes
CAPACITY = {"Road": 26, "Rail": 16, "Air": 10, "Sea": 8}   # max shipments/month per mode

# ---------------------------------------------------------------
# 3. Baseline ("current") allocation - proportional to overall mode mix
# ---------------------------------------------------------------
mode_share = df["transport_mode"].value_counts(normalize=True).reindex(modes)
current_alloc = np.round(mode_share.values * TOTAL_SHIPMENTS)
# adjust rounding so it sums exactly to TOTAL_SHIPMENTS
current_alloc[np.argmax(current_alloc)] += TOTAL_SHIPMENTS - current_alloc.sum()

current_cost = float(np.dot(current_alloc, cost))
current_delay = float(np.dot(current_alloc, delay) / TOTAL_SHIPMENTS)

# SLA constraint for the optimizer: match today's average service level
# (the goal is to find a CHEAPER mix that is no slower than today, not to
# further tighten the delay target)
SLA_TARGET_HRS = round(current_delay, 2)

print(f"\nCurrent mode mix (proportional to historical usage):")
for m, a in zip(modes, current_alloc):
    print(f"  {m:5s}: {int(a)} shipments/month")
print(f"Current total cost: Rs {current_cost:,.0f}/month | Current avg delay: {current_delay:.2f} hrs")
print(f"SLA constraint used for optimization: avg delay <= {SLA_TARGET_HRS} hrs (i.e., no worse than today)")

# ---------------------------------------------------------------
# 4. Linear Program: minimize cost s.t. capacity + SLA constraints
#    Decision variables: x_mode = number of shipments assigned to each mode
#    minimize   sum(cost_m * x_m)
#    s.t.       sum(x_m) = TOTAL_SHIPMENTS               (equality)
#               x_m <= CAPACITY[m]                        (upper bound)
#               sum(delay_m * x_m) <= SLA_TARGET_HRS * TOTAL_SHIPMENTS  (SLA)
#               x_m >= 0
# ---------------------------------------------------------------
c = cost  # objective coefficients (minimize total cost)

A_eq = [np.ones(len(modes))]
b_eq = [TOTAL_SHIPMENTS]

A_ub = [delay]  # sum(delay_m * x_m) <= SLA_TARGET_HRS * TOTAL_SHIPMENTS
b_ub = [SLA_TARGET_HRS * TOTAL_SHIPMENTS]

bounds = [(0, CAPACITY[m]) for m in modes]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

if result.success:
    optimized_alloc = np.round(result.x, 1)
    optimized_cost = float(np.dot(optimized_alloc, cost))
    optimized_delay = float(np.dot(optimized_alloc, delay) / TOTAL_SHIPMENTS)

    print("\n" + "="*70)
    print("OPTIMIZATION RESULT (Linear Programming, scipy.optimize.linprog)")
    print("="*70)
    for m, a in zip(modes, optimized_alloc):
        print(f"  {m:5s}: {a} shipments/month")
    print(f"Optimized total cost: Rs {optimized_cost:,.0f}/month | Optimized avg delay: {optimized_delay:.2f} hrs")

    savings = current_cost - optimized_cost
    savings_pct = savings / current_cost * 100
    print(f"\nMonthly cost saving vs. current mix: Rs {savings:,.0f} ({savings_pct:.1f}%)")
    print(f"Annualized saving: Rs {savings*12:,.0f}")
else:
    print("Optimization failed:", result.message)

# ---------------------------------------------------------------
# 5. Save results & chart
# ---------------------------------------------------------------
opt_summary = pd.DataFrame({
    "Mode": modes,
    "Current_Allocation": current_alloc.astype(int),
    "Optimized_Allocation": optimized_alloc,
    "Cost_per_Shipment": np.round(cost, 0),
    "Avg_Delay_hrs": np.round(delay, 2),
})
opt_summary.to_csv(BASE_DIR / "optimization_result.csv", index=False)
print("\n", opt_summary.to_string(index=False))

# Chart: current vs optimized allocation by mode
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(modes))
width = 0.35
ax.bar(x - width/2, current_alloc, width, label="Current Mix", color=PALETTE[0])
ax.bar(x + width/2, optimized_alloc, width, label="Optimized Mix", color=PALETTE[2])
ax.set_xticks(x)
ax.set_xticklabels(modes)
ax.set_ylabel("Shipments / Month")
ax.set_title(f"Mode Allocation: Current vs. Optimized\n(Lane: {busiest_lane})", fontsize=13, weight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/06_optimization_allocation.png", dpi=150)
plt.show()

# Chart: cost & delay before/after comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].bar(["Current", "Optimized"], [current_cost, optimized_cost], color=[PALETTE[0], PALETTE[2]])
axes[0].set_title("Total Monthly Cost (Rs)")
for i, v in enumerate([current_cost, optimized_cost]):
    axes[0].text(i, v + 3000, f"Rs {v:,.0f}", ha="center", fontsize=10)
axes[1].bar(["Current", "Optimized"], [current_delay, optimized_delay], color=[PALETTE[0], PALETTE[2]])
axes[1].axhline(SLA_TARGET_HRS, color=PALETTE[3], linestyle="--", label=f"SLA target ({SLA_TARGET_HRS} hrs)")
axes[1].set_title("Avg. Predicted Delay (hrs)")
axes[1].legend()
for i, v in enumerate([current_delay, optimized_delay]):
    axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
plt.suptitle("Optimization Impact: Cost & Delay", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/07_optimization_cost_delay_impact.png", dpi=150)
plt.show()
print("\nOptimization charts saved to:")
print(f"  {OUT / '06_optimization_allocation.png'}")
print(f"  {OUT / '07_optimization_cost_delay_impact.png'}")
