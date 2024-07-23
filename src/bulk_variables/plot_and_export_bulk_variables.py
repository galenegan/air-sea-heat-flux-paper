import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import get_project_root
from src.bulk_variables.bulk_models import (
    get_train_val_test,
    incoming_shortwave_random_forest,
    air_temperature_nn,
    specific_humidity_nn,
)

params = {
    "axes.labelsize": 20,
    "font.size": 16,
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
df = incoming_shortwave_random_forest(df)
df = air_temperature_nn(df)
df = specific_humidity_nn(df)

# Exporting for bulk flux calculations
df.to_csv(f"{project_root}/data/bulk_variable_dataset.csv", index=False)

# %% Plotting
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)


# %% Air temperature
predicted_variable = "estimated_air_temperature_nn"
ground_truth_variable = "air_temperature_surface"
train_score = mae(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_score = mae(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_score = mae(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])
train_bias = bias(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_bias = bias(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_bias = bias(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])

one_train = np.linspace(
    df.loc[train_idx, ground_truth_variable].min(), df.loc[train_idx, ground_truth_variable].max(), 100
)
one_val = np.linspace(df.loc[val_idx, ground_truth_variable].min(), df.loc[val_idx, ground_truth_variable].max(), 100)
one_test = np.linspace(
    df.loc[test_idx, ground_truth_variable].min(), df.loc[test_idx, ground_truth_variable].max(), 100
)

lower_lim = -6
upper_lim = 1.2 * np.max(df[[ground_truth_variable, predicted_variable]])

ax1.plot(
    df.loc[train_idx, predicted_variable],
    df.loc[train_idx, ground_truth_variable],
    "o",
    alpha=0.5,
    color="#012749",
    markersize=4,
)
ax1.annotate(
    f"Train: MAE = {train_score:.2f}" + r"$^\circ$C," + f" Bias = {train_bias:.2f}" + r"$^\circ$C",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
)
ax1.plot(one_train, one_train, "-", color="#9f1853", linewidth=3)

ax2.plot(
    df.loc[val_idx, predicted_variable],
    df.loc[val_idx, ground_truth_variable],
    "o",
    alpha=0.5,
    color="#012749",
    markersize=4,
)
ax2.plot(one_val, one_val, "-", color="#9f1853", linewidth=3)
ax2.annotate(
    f"Val: MAE = {eval_score:.2f}" + r"$^\circ$C," + f" Bias = {eval_bias:.2f}" + r"$^\circ$C",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
)

ax3.plot(
    df.loc[test_idx, predicted_variable],
    df.loc[test_idx, ground_truth_variable],
    "o",
    alpha=0.5,
    color="#012749",
    markersize=4,
)
ax3.plot(one_test, one_test, "-", color="#9f1853", linewidth=3)
ax3.annotate(
    f"Test: MAE = {test_score:.2f}" + r"$^\circ$C," + f" Bias = {test_bias:.2f}" + r"$^\circ$C",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
)

ax1.set_title("(d)")
ax2.set_title("(e)")
ax3.set_title("(f)")

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel(r"Spotter $T_{\textrm{air}}$ $(^\circ C)$")
    ax.set_ylabel(r"ASIT $T_{\textrm{air}}$ $(^\circ C)$")
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_xticks(np.arange(-5, 31, 5))

print(test_score / np.ptp(df.loc[test_idx, ground_truth_variable]))


# %% Shortwave
predicted_variable = "estimated_shortwave_rf"
ground_truth_variable = "solar_down"
train_score = mae(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_score = mae(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_score = mae(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])
train_bias = bias(df.loc[train_idx, predicted_variable], df.loc[train_idx, ground_truth_variable])
eval_bias = bias(df.loc[val_idx, predicted_variable], df.loc[val_idx, ground_truth_variable])
test_bias = bias(df.loc[test_idx, predicted_variable], df.loc[test_idx, ground_truth_variable])

one_train = np.linspace(
    df.loc[train_idx, ground_truth_variable].min(), df.loc[train_idx, ground_truth_variable].max(), 100
)
one_val = np.linspace(df.loc[val_idx, ground_truth_variable].min(), df.loc[val_idx, ground_truth_variable].max(), 100)
one_test = np.linspace(
    df.loc[test_idx, ground_truth_variable].min(), df.loc[test_idx, ground_truth_variable].max(), 100
)

lower_lim = -100
upper_lim = 1.15 * np.max(df[[ground_truth_variable, predicted_variable]])
ax4.plot(
    df.loc[train_idx, predicted_variable],
    df.loc[train_idx, ground_truth_variable],
    "o",
    alpha=0.5,
    color="#012749",
    markersize=4,
)
ax4.annotate(
    f"Train: MAE = {train_score:.2f}" + r" W/m$^2$," + f" Bias = {train_bias:.2f}" + f" W/m$^2$",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
)
ax4.plot(one_train, one_train, "-", color="#9f1853", linewidth=3)

ax5.plot(
    df.loc[val_idx, predicted_variable],
    df.loc[val_idx, ground_truth_variable],
    "o",
    alpha=0.5,
    color="#012749",
    markersize=4,
)
ax5.plot(one_val, one_val, "-", color="#9f1853", linewidth=3)
ax5.annotate(
    f"Val: MAE = {eval_score:.2f}" + r" W/m$^2$," + f" Bias = {eval_bias:.2f}" + f" W/m$^2$",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
)

ax6.plot(
    df.loc[test_idx, predicted_variable],
    df.loc[test_idx, ground_truth_variable],
    "o",
    alpha=0.5,
    color="#012749",
    markersize=4,
)
ax6.plot(one_test, one_test, "-", color="#9f1853", linewidth=3)
ax6.annotate(
    f"Test: MAE = {test_score:.2f}" + r" W/m$^2$," + f" Bias = {test_bias:.2f}" + f" W/m$^2$",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
)

ax4.set_title("(a)")
ax5.set_title("(b)")
ax6.set_title("(c)")

for ax in [ax4, ax5, ax6]:
    ax.set_xlabel(r"Spotter $Q_{SW,\textrm{down}}$ (W/m$^2$)")
    ax.set_ylabel(r"ASIT $Q_{SW,\textrm{down}}$ (W/m$^2$)")
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)

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

one_train = np.linspace(
    df.loc[train_idx, ground_truth_variable].min(), df.loc[train_idx, ground_truth_variable].max(), 100
)
one_val = np.linspace(df.loc[val_idx, ground_truth_variable].min(), df.loc[val_idx, ground_truth_variable].max(), 100)
one_test = np.linspace(
    df.loc[test_idx, ground_truth_variable].min(), df.loc[test_idx, ground_truth_variable].max(), 100
)

lower_lim = 1
upper_lim = 1.15 * np.max(df[[ground_truth_variable, predicted_variable]])

print(test_score / np.ptp(df.loc[test_idx, ground_truth_variable]))

ax7.plot(
    df.loc[train_idx, predicted_variable],
    df.loc[train_idx, ground_truth_variable],
    "o",
    alpha=0.5,
    color="#012749",
    markersize=4,
)
ax7.annotate(
    f"Train: MAE = {train_score:.2f}" + " g/kg," + f" Bias = {train_bias:.2f}" + " g/kg",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
)
ax7.plot(one_train, one_train, "-", color="#9f1853", linewidth=3)

ax8.plot(
    df.loc[val_idx, predicted_variable],
    df.loc[val_idx, ground_truth_variable],
    "o",
    alpha=0.5,
    color="#012749",
    markersize=4,
)
ax8.plot(one_val, one_val, "-", color="#9f1853", linewidth=3)
ax8.annotate(
    f"Val: MAE = {eval_score:.2f}" + " g/kg," + f" Bias = {eval_bias:.2f}" + " g/kg",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
)

ax9.plot(
    df.loc[test_idx, predicted_variable],
    df.loc[test_idx, ground_truth_variable],
    "o",
    alpha=0.5,
    color="#012749",
    markersize=4,
)
ax9.plot(one_test, one_test, "-", color="#9f1853", linewidth=3)
ax9.annotate(
    f"Test: MAE = {test_score:.2f}" + " g/kg," + f" Bias = {test_bias:.2f}" + " g/kg",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
)


ax7.set_title("(g)")
ax8.set_title("(h)")
ax9.set_title("(i)")

for ax in [ax7, ax8, ax9]:
    ax.set_xlabel(r"Spotter $q$ (g/kg)")
    ax.set_ylabel(r"ASIT $q$ (g/kg)")
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)

fig.set_size_inches(19, 15)
fig.tight_layout(pad=0.5)
plt.savefig(f"{project_root}/plots/bulk_variables.png", dpi=300)
plt.show()
