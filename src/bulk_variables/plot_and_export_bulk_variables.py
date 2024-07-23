import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import get_project_root
from src.bulk_variables.bulk_models import (
    get_train_val_test,
    incoming_shortwave,
    air_temperature_nn,
    specific_humidity_nn,
)

params = {
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 13,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "axes.grid": False,
}
plt.rcParams.update(params)


def mae(x, y):
    return np.nanmean(np.abs(x - y))


def bias(x, y):
    return np.nanmean(x - y)


project_root = get_project_root()
df = pd.read_csv(f"{project_root}/data/training_dataset.csv")
spot_ids_vented = ["31081C", "31084C", "31085C"]
df = df.loc[df["spot_id"].isin(spot_ids_vented)]
df = df.reset_index(drop=True)

# Train test split by time
train_idx, val_idx, test_idx = get_train_val_test(df)
full_idx = train_idx.union(val_idx.union(test_idx))

# %% Estimating the bulk variables
df["inferred_solar_radiation"] = incoming_shortwave(df)
df["estimated_air_temperature_nn"] = air_temperature_nn(df)
df["estimated_specific_humidity_nn"] = specific_humidity_nn(df)

# Exporting for bulk flux calculations
df.to_csv(f"{project_root}/data/bulk_variable_dataset.csv", index=False)

# %% Semi-analytical air temperature
dfplot = df.loc[full_idx]
mae_plot = np.nanmean(np.abs(dfplot["estimated_air_temperature"].values - dfplot["air_temperature_surface"].values))
bias_plot = np.nanmean(dfplot["estimated_air_temperature"].values - dfplot["air_temperature_surface"].values)
one = np.linspace(dfplot["air_temperature_surface"].min(), dfplot["air_temperature_surface"].max(), 100)

fig, ax = plt.subplots()
ax.plot(
    dfplot["estimated_air_temperature"].values,
    dfplot["air_temperature_surface"].values,
    "o",
    color="#012749",
    alpha=0.5,
)
ax.plot(
    one,
    one,
    "-",
    color="#9f1853",
    linewidth=3,
    label=f"MAE = {mae_plot:.2f} " + r"$^\circ$C" + f", Bias = {bias_plot:.2f} " + r"$^\circ$C",
)
ax.legend()
ax.set_xlabel(r"Analytical $\tilde{T}_{\textrm{air}}$ $(^\circ C)$")
ax.set_ylabel(r"ASIT $T_{\textrm{air}}$ $(^\circ C)$")
ax.set_xlim(-6, 35)
ax.set_ylim(-6, 35)
ax.set_xticks(np.arange(-5, 36, 5))
ax.set_yticks(np.arange(-5, 36, 5))
fig.set_size_inches(7, 6)
fig.tight_layout(pad=0.5)
plt.rcParams.update(params)
# plt.savefig(f"{project_root}/plots/tair_analytical.png", dpi=300)
plt.show()

# %% Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Shortwave first
predicted_variable = "inferred_solar_radiation"
ground_truth_variable = "solar_down_24m"
train_score = mae(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_score = mae(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_score = mae(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])
train_bias = bias(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_bias = bias(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_bias = bias(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])
one = np.linspace(df.loc[full_idx, ground_truth_variable].min(), df.loc[full_idx, ground_truth_variable].max(), 100)
lower_lim = -100
upper_lim = 1.4 * np.max(df[[ground_truth_variable, predicted_variable]])
ax1.plot(
    df.loc[train_idx, predicted_variable],
    df.loc[train_idx, ground_truth_variable],
    "o",
    alpha=0.8,
    color="#009d9a",
    markersize=4,
    label=f"Train, MAE = {train_score:.2f}" + r" W/m$^2$," + f" Bias = {train_bias:.2f}" + f" W/m$^2$",
)
ax1.plot(
    df.loc[val_idx, predicted_variable],
    df.loc[val_idx, ground_truth_variable],
    "o",
    alpha=0.4,
    color="#002d9c",
    markersize=6,
    label=f"Val, MAE = {eval_score:.2f}" + r" W/m$^2$," + f" Bias = {eval_bias:.2f}" + f" W/m$^2$",
)
ax1.plot(
    df.loc[test_idx, predicted_variable],
    df.loc[test_idx, ground_truth_variable],
    "o",
    alpha=0.4,
    color="#a56eff",
    markersize=6,
    label=f"Test, MAE = {test_score:.2f}" + r" W/m$^2$," + f" Bias = {test_bias:.2f}" + f" W/m$^2$",
)
ax1.plot(one, one, "-", color="#9f1853", linewidth=3)

ax1.set_title("(a)")
ax1.set_xlabel(r"Spotter $Q_{SW,\textrm{down}}$ (W/m$^2$)")
ax1.set_ylabel(r"ASIT $Q_{SW,\textrm{down}}$ (W/m$^2$)")
ax1.legend(loc="upper left")
ax1.set_xlim(lower_lim, upper_lim)
ax1.set_ylim(lower_lim, upper_lim)

print(test_score / np.ptp(df.loc[test_idx, ground_truth_variable]))

# %% Air temperature
predicted_variable = "estimated_air_temperature_nn"
ground_truth_variable = "air_temperature_surface"
train_score = mae(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_score = mae(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_score = mae(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])
train_bias = bias(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_bias = bias(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_bias = bias(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])
one = np.linspace(df.loc[full_idx, ground_truth_variable].min(), df.loc[full_idx, ground_truth_variable].max(), 100)
lower_lim = -2.5
upper_lim = 1.3 * np.max(df[[ground_truth_variable, predicted_variable]])


ax2.plot(
    df.loc[train_idx, predicted_variable],
    df.loc[train_idx, ground_truth_variable],
    "o",
    alpha=0.8,
    color="#009d9a",
    markersize=4,
    label=f"Train, MAE = {train_score:.2f} " + r"$^\circ$C," + f" Bias = {train_bias:.2f} " + r"$^\circ$C",
)
ax2.plot(
    df.loc[val_idx, predicted_variable],
    df.loc[val_idx, ground_truth_variable],
    "o",
    alpha=0.2,
    color="#002d9c",
    markersize=6,
    label=f"Val, MAE = {eval_score:.2f} " + r"$^\circ$C," + f" Bias = {eval_bias:.2f} " + r"$^\circ$C",
)
ax2.plot(
    df.loc[test_idx, predicted_variable],
    df.loc[test_idx, ground_truth_variable],
    "o",
    alpha=0.2,
    color="#a56eff",
    markersize=6,
    label=f"Test, MAE = {test_score:.2f} " + r"$^\circ$C," + f" Bias = {test_bias:.2f} " + r"$^\circ$C",
)
ax2.plot(one, one, "-", color="#9f1853", linewidth=3)
ax2.set_title("(b)")
ax2.set_xlabel(r"Spotter $T_{\textrm{air}}$ $(^\circ C)$")
ax2.set_ylabel(r"ASIT $T_{\textrm{air}}$ $(^\circ C)$")
leg = ax2.legend(loc="upper left")
for lh in leg.legendHandles[1:]:
    lh.set_alpha(0.4)
ax2.set_xlim(lower_lim, upper_lim)
ax2.set_ylim(lower_lim, upper_lim)

print(test_score / np.ptp(df.loc[test_idx, ground_truth_variable]))

# %% Specific Humidity
predicted_variable = "estimated_specific_humidity_nn"
ground_truth_variable = "specific_humidity_surface"
train_score = mae(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_score = mae(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_score = mae(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])
train_bias = bias(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_bias = bias(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_bias = bias(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])
one = np.linspace(df.loc[full_idx, ground_truth_variable].min(), df.loc[full_idx, ground_truth_variable].max(), 100)
lower_lim = 1
upper_lim = 1.25 * np.max(df[[ground_truth_variable, predicted_variable]])

print(test_score / np.ptp(df.loc[test_idx, ground_truth_variable]))

ax3.plot(
    df.loc[train_idx, predicted_variable],
    df.loc[train_idx, ground_truth_variable],
    "o",
    alpha=0.8,
    color="#009d9a",
    markersize=4,
    label=f"Train, MAE = {train_score:.2f} g/kg," + f" Bias = {train_bias:.2f} g/kg",
)
ax3.plot(
    df.loc[val_idx, predicted_variable],
    df.loc[val_idx, ground_truth_variable],
    "o",
    alpha=0.2,
    color="#002d9c",
    markersize=6,
    label=f"Val, MAE = {eval_score:.2f} g/kg," + f" Bias = {eval_bias:.2f} g/kg",
)
ax3.plot(
    df.loc[test_idx, predicted_variable],
    df.loc[test_idx, ground_truth_variable],
    "o",
    alpha=0.2,
    color="#a56eff",
    markersize=6,
    label=f"Test, MAE = {test_score:.2f} g/kg," + f" Bias = {test_bias:.2f} g/kg",
)
ax3.plot(one, one, "-", color="#9f1853", linewidth=3)
ax3.set_title("(c)")
ax3.set_xlabel(r"Spotter $q$ (g/kg)")
ax3.set_ylabel(r"ASIT $q$ (g/kg)")
leg = ax3.legend(loc="upper left")
for lh in leg.legendHandles[1:]:
    lh.set_alpha(0.4)
ax3.set_xlim(lower_lim, upper_lim)
ax3.set_ylim(lower_lim, upper_lim)

fig.set_size_inches(17, 5)
fig.tight_layout(pad=0.5)
plt.savefig(f"{project_root}/plots/bulk_variables.png", dpi=300)
plt.show()
