import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("=== Groundwater Check: Multi-State (Stable Sample) ===")

# ---------- 1) Build the dataset ----------
def make_dates(start, days):
    """Return a list of datetime stamps from `start` over `days` consecutive days."""
    return [start + timedelta(days=i) for i in range(days)]

# Dates: 10 consecutive days from 1 Oct 2025
dates = make_dates(datetime(2025, 10, 1), 10)

# ----- FIXED STATES -----
states = (
    ['Rajasthan'] * 10 +
    ['Gujarat'] * 10 +
    ['Punjab'] * 10 +
    ['Tamil Nadu'] * 10 +
    ['Meghalaya'] * 10       # <-- Added Meghalaya (fixed)
)

# ----- FIXED LOCATIONS -----
sites = (
    ['Jaipur - Well A'] * 10 +
    ['Ahmedabad - Well B'] * 10 +
    ['Ludhiana - Well C'] * 10 +
    ['Coimbatore - Well D'] * 10 +
    ['Shillong - Well E'] * 10   # <-- Added location for Meghalaya
)

# ----- FIXED WATER LEVEL DATA -----
levels = [
    # Rajasthan (Jaipur)
    14.2, 14.18, 14.15, 14.10, 14.05, 13.98, 13.90, 13.80, 13.68, 13.55,
    
    # Gujarat (Ahmedabad)
    11.8, 11.79, 11.75, 11.73, 11.70, 11.67, 11.63, 11.58, 11.50, 11.42,

    # Punjab (Ludhiana)
    8.4, 8.38, 8.36, 8.34, 8.31, 8.28, 8.24, 8.20, 8.15, 8.10,

    # Tamil Nadu (Coimbatore)
    16.5, 16.48, 16.46, 16.43, 16.40, 16.36, 16.32, 16.25, 16.18, 16.10,

    # Meghalaya (Shillong)  <-- NEW DATA (slight depletion pattern)
    9.2, 9.18, 9.17, 9.15, 9.13, 9.10, 9.08, 9.05, 9.02, 8.98,
]

# Build DataFrame
df = pd.DataFrame({
    "timestamp": dates * 5,  # now 5 states
    "state": states,
    "location": sites,
    "water_level_m": levels
})

print("\nData ready ✓  (showing a quick peek)")
print(df.head(15))

# ---------- 2) Trend lines by state ----------
plt.figure(figsize=(10, 6))
for state_name, group in df.groupby("state"):
    plt.plot(group["timestamp"], group["water_level_m"], marker="o", label=state_name)

plt.title("Groundwater Level Trend Across States")
plt.xlabel("Date")
plt.ylabel("Water Level (m)")
plt.legend(title="State")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- 3) Depletion vs Day 1 (per state) ----------
agg = (
    df.sort_values(["state", "timestamp"])
      .groupby("state")["water_level_m"]
      .agg(first="first", last="last")
)
agg["depletion_m"] = agg["first"] - agg["last"]

df = df.merge(agg["depletion_m"], left_on="state", right_index=True, how="left")

depletion_summary = agg.reset_index()[["state", "depletion_m"]]

print("\n----- State-wise Depletion Summary -----")
print(depletion_summary)

# ---------- 4) Bar chart: total depletion by state ----------
plt.figure(figsize=(8, 5))
plt.bar(depletion_summary["state"], depletion_summary["depletion_m"])
plt.title("Total Groundwater Level Depletion by State (Over the Period)")
plt.xlabel("State")
plt.ylabel("Depletion (m)")
plt.grid(True)
plt.tight_layout()
plt.show()
