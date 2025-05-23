"""
Generates Figure 10b
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from src.utils import get_project_root

params = {
    "axes.labelsize": 16,
    "font.size": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "axes.grid": False,
}
plt.rcParams.update(params)


def fill_night_temp(df):
    dfout = pd.DataFrame()
    for spot_id in df["spot_id"].unique():
        dfs = df.loc[df["spot_id"] == spot_id]
        dfs["night_temp"] = dfs.loc[dfs["solar_voltage"] < 1, "air_temperature"]
        dfs["night_sst"] = dfs.loc[dfs["solar_voltage"] < 1, "sea_surface_temperature"]
        dfs["night_temp_interp"] = dfs["night_temp"].interpolate(method="ffill")
        dfs["night_sst_interp"] = dfs["night_sst"].interpolate(method="ffill")
        dfs["delta_night_temp"] = dfs["air_temperature"] - dfs["night_temp_interp"]
        dfs["delta_night_sst"] = dfs["sea_surface_temperature"] - dfs["night_sst_interp"]
        dfs["delta_in_out"] = dfs["air_temperature"] - dfs["air_temperature_surface"]
        dfs["delta_air_water"] = dfs["air_temperature"] - dfs["sea_surface_temperature"]
        dfs["temp_derivative"] = np.gradient(dfs["air_temperature"], dfs["epoch"])
        dfout = pd.concat([dfout, dfs])
    return dfout


project_root = get_project_root()
data_path = f"{project_root}/data"
df = pd.read_csv(f"{data_path}/training_dataset.csv")
spot_ids_unvented = ["31080C", "31082C"]

df = df.loc[df["spot_id"].isin(spot_ids_unvented)]
df = fill_night_temp(df)
df = df.sort_values(by="time")
df = df.dropna(subset=["delta_night_temp", "delta_in_out"])

df_train = df.loc[df["time"] >= "2024-01-01"]
df_test = df.loc[df["time"] < "2024-01-01"]


# %% Making the prediction
X_train = np.vstack((df_train["delta_night_temp"].values, df_train["delta_night_sst"].values)).T
y_train = df_train["delta_in_out"].values
one_train = np.linspace(df_train["air_temperature_surface"].min(), df_train["air_temperature_surface"].max(), 100)

X_test = np.vstack((df_test["delta_night_temp"].values, df_test["delta_night_sst"].values)).T
y_test = df_test["delta_in_out"].values
one_test = np.linspace(df_test["air_temperature_surface"].min(), df_test["air_temperature_surface"].max(), 100)

X = np.vstack((df["delta_night_temp"].values, df["delta_night_sst"].values)).T
model = LinearRegression().fit(X_train, y_train)

y_pred_train = df_train["air_temperature"].values - model.predict(X_train)
y_pred_test = df_test["air_temperature"].values - model.predict(X_test)
mae_train = np.nanmean(np.abs(y_pred_train - df_train["air_temperature_surface"]))
mae_test = np.nanmean(np.abs(y_pred_test - df_test["air_temperature_surface"]))

df["estimated_air_temperature"] = df["air_temperature"].values - model.predict(X)
# df_vented.to_csv(f"{data_path}/specific_humidity_training_dataset_vented.csv", index=False)
# %% Plotting
mae_plot = np.nanmean(np.abs(df["estimated_air_temperature"].values - df["air_temperature_surface"].values))
rmse_plot = np.sqrt(np.nanmean((df["estimated_air_temperature"].values - df["air_temperature_surface"].values) ** 2))
bias_plot = np.nanmean(df["estimated_air_temperature"].values - df["air_temperature_surface"].values)
one = np.linspace(df["air_temperature_surface"].min(), df["air_temperature_surface"].max(), 100)

fig, ax1 = plt.subplots()
ax1.plot(
    df["estimated_air_temperature"].values,
    df["air_temperature_surface"].values,
    "o",
    color="#012749",
    alpha=0.5,
)
ax1.plot(
    one,
    one,
    "-",
    color="#9f1853",
    linewidth=3,
    label=f"MAE = {mae_plot:.2f}, RMSE = {rmse_plot:.2f}, Bias = {bias_plot:.2f} " + r"$^\circ$C",
)
ax1.legend()
ax1.set_xlabel(r"Linear Regression $\tilde{T}_{\textrm{air}}$ $(^\circ C)$")
ax1.set_ylabel(r"ASIT $T_{\textrm{air}}$ $(^\circ C)$")
ax1.set_xlim(-6, 32)
ax1.set_ylim(-6, 32)
ax1.set_xticks(np.arange(-5, 31, 5))
ax1.set_yticks(np.arange(-5, 31, 5))
fig.set_size_inches(6, 5)
fig.tight_layout(pad=0.5)
# plt.savefig(f"{project_root}/plots/unvented_tair.png", dpi=300)
plt.show()