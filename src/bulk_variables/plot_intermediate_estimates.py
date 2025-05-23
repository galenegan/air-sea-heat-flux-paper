"""
Generates Figure 3b, 4b
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import get_project_root
from src.bulk_variables.bulk_models import (
    air_temperature_linear,
    incoming_shortwave_box_model,
    specific_humidity_linear,
    get_train_val_test,
)

params = {
    "axes.labelsize": 20,
    "font.size": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "axes.grid": False,
}
plt.rcParams.update(params)


def mae(predicted, actual):
    return np.nanmean(np.abs(predicted - actual))


def rmse(predicted, actual):
    return np.sqrt(np.nanmean((predicted - actual) ** 2))


def bias(predicted, actual):
    return np.nanmean(predicted - actual)


project_root = get_project_root()
df = pd.read_csv(f"{project_root}/data/training_dataset.csv")
spot_ids_vented = ["31081C", "31084C", "31085C"]
df = df.loc[df["spot_id"].isin(spot_ids_vented)]
train, val, test = get_train_val_test(df)
df = df.loc[test]
df = df.reset_index(drop=True)

df = air_temperature_linear(df)
df = incoming_shortwave_box_model(df)
df = specific_humidity_linear(df)

# %% Plotting the three variables
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Air temperature
mae_plot = np.nanmean(np.abs(df["estimated_air_temperature"].values - df["air_temperature_surface"].values))
bias_plot = np.nanmean(df["estimated_air_temperature"].values - df["air_temperature_surface"].values)
rmse_plot = rmse(df["estimated_air_temperature"].values, df["air_temperature_surface"].values)
one = np.linspace(df["air_temperature_surface"].min(), df["air_temperature_surface"].max(), 100)

ax1.plot(
    df["estimated_air_temperature"].values,
    df["air_temperature_surface"].values,
    "o",
    color="#012749",
    alpha=0.5,
)
ax1.plot(one, one, "-", color="#9f1853", linewidth=3)
ax1.annotate(
    f"MAE = {mae_plot:.2f}, RMSE = {rmse_plot:.2f}, Bias = {bias_plot:.2f}" + r" $^\circ$C",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
    fontsize=14,
)
ax1.set_xlabel(r"Linear Regression $\tilde{T}_{\textrm{air}}$ $(^\circ C)$")
ax1.set_ylabel(r"ASIT $T_{\textrm{air}}$ $(^\circ C)$")
ax1.set_xlim(-6, 32)
ax1.set_ylim(-6, 32)
ax1.set_xticks(np.arange(-5, 31, 5))
ax1.set_yticks(np.arange(-5, 31, 5))


# Shortwave
mae_plot = np.nanmean(np.abs(df["box_model_solar"].values - df["solar_down"].values))
rmse_plot = rmse(df["box_model_solar"].values, df["solar_down"].values)
bias_plot = np.nanmean(df["box_model_solar"].values - df["solar_down"].values)
one = np.linspace(df["solar_down"].min(), df["solar_down"].max(), 100)

ax2.plot(
    df["box_model_solar"].values,
    df["solar_down"].values,
    "o",
    color="#012749",
    alpha=0.5,
)
ax2.plot(one, one, "-", color="#9f1853", linewidth=3)
ax2.annotate(
    f"MAE = {mae_plot:.2f}, RMSE = {rmse_plot:.2f}, Bias = {bias_plot:.2f}" + r" W/m$^2$",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
    fontsize=14,
)
ax2.set_xlabel(r"Box Model $\tilde{Q}_{SW,\textrm{down}}$ (W/m$^2$)")
ax2.set_ylabel(r"ASIT $Q_{SW,\textrm{down}}$ (W/m$^2$)")
ax2.set_xlim(-100, 1250)
ax2.set_ylim(-100, 1250)
ax2.set_xticks(np.arange(0, 1250, 200))
ax2.set_yticks(np.arange(0, 1250, 200))

# Specific humidity
mae_plot = np.nanmean(np.abs(df["estimated_specific_humidity"].values - df["specific_humidity_surface"].values))
rmse_plot = rmse(df["estimated_specific_humidity"].values, df["specific_humidity_surface"].values)
bias_plot = np.nanmean(df["estimated_specific_humidity"].values - df["specific_humidity_surface"].values)
one = np.linspace(df["specific_humidity_surface"].min(), df["specific_humidity_surface"].max(), 100)

ax3.plot(
    df["estimated_specific_humidity"].values,
    df["specific_humidity_surface"].values,
    "o",
    color="#012749",
    alpha=0.5,
)
ax3.plot(one, one, "-", color="#9f1853", linewidth=3)
ax3.annotate(
    f"MAE = {mae_plot:.2f}, RMSE = {rmse_plot:.2f}, Bias = {bias_plot:.2f}" + " g/kg",
    xy=(0.02, 0.9),
    xycoords="axes fraction",
    fontsize=14,
)
ax3.set_xlabel(r"Linear Model $\tilde{q}$ (g/kg)")
ax3.set_ylabel(r"ASIT $q$ (g/kg)")
ax3.set_xlim(1, 21.34)
ax3.set_ylim(1, 21.34)
ax3.set_xticks(np.arange(5, 21, 5))
ax3.set_yticks(np.arange(5, 21, 5))

fig.set_size_inches(17, 5)
fig.tight_layout(pad=0.5)
plt.rcParams.update(params)
# plt.savefig(f"{project_root}/plots/intermediate_bulk_variables.png", dpi=600)
plt.show()
