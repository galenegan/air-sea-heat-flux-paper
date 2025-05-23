import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import get_project_root
from src.bulk_variables.bulk_models import (
    air_temperature_linear,
    incoming_shortwave_box_model,
    specific_humidity_exp_decay,
    get_train_val_test,
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
df = specific_humidity_exp_decay(df)

df["qres"] = df["estimated_q_outer"] - df["specific_humidity_surface"]
df["time"] = pd.to_datetime(df["time"], utc=True)
# %% Sandbox plots
istart = 2000
iend = 3000
# time series
fig, ax = plt.subplots()
ax.plot(df["time"].iloc[istart:iend], df["specific_humidity_surface"].iloc[istart:iend], label="actual")
ax.plot(df["time"].iloc[istart:iend], df["estimated_q_outer"].iloc[istart:iend], label="predicted")
fig.autofmt_xdate()
ax.legend()
plt.show()

# %% Scatter with qsw
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(df["estimated_q_outer"], df["specific_humidity_surface"], c=df["delta_night_air_temp"], cmap="viridis")
cb = plt.colorbar(sc, ax=ax)
ax.set_xlabel(r"Estimated q")
ax.set_ylabel(r"Actual q")
cb.set_label(r"internal temp")
fig.tight_layout(pad=0.5)
plt.show()

# %% Scatter with qsw
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    df.loc[df["solar_voltage"] < 1, "estimated_q_outer"], df.loc[df["solar_voltage"] < 1, "specific_humidity_surface"]
)
ax.set_xlabel(r"Estimated q")
ax.set_ylabel(r"Actual q")
fig.tight_layout(pad=0.5)
plt.show()

# %% Big pairplot
import seaborn as sns

df = df.loc[df["dTdt_filt"] < 0.0025]
df["delta_t"] = df["air_temperature"] - df["estimated_air_temperature"]
columns = ["qres", "estimated_air_temperature", "atmospheric_pressure", "delta_t"]
fig = plt.figure()
sns.pairplot(data=df, vars=columns, diag_kind=None)
plt.show()

# %% linear regression
from sklearn.linear_model import LinearRegression

dflr = df.dropna(
    subset=["estimated_air_temperature", "atmospheric_pressure", "delta_t", "qres", "significant_wave_height"]
)
X = dflr[["estimated_air_temperature", "atmospheric_pressure", "delta_t"]]
y = dflr["qres"]

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

q_pred = dflr["estimated_q_outer"] - y_pred

fig = plt.figure()
plt.scatter(q_pred, dflr["specific_humidity_surface"])
plt.show()

fig = plt.figure()
plt.scatter(dflr["estimated_q_outer"], dflr["specific_humidity_surface"])
plt.show()
