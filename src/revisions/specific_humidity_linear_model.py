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
from sklearn.linear_model import LinearRegression

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
df = air_temperature_linear(df)
df = incoming_shortwave_box_model(df)
df = specific_humidity_exp_decay(df)

df["qres"] = df["estimated_q_outer"] - df["specific_humidity_surface"]
df["delta_t"] = df["air_temperature"] - df["estimated_air_temperature"]

# %% Training the model
dflr = df.dropna(subset=["estimated_air_temperature", "atmospheric_pressure", "dTdt_filt", "qres"])
dflr = dflr.reset_index(drop=True)
train, val, test = get_train_val_test(dflr)

X_train = dflr.loc[train.union(val), ["estimated_air_temperature", "atmospheric_pressure", "dTdt_filt"]]
X_test = dflr.loc[test, ["estimated_air_temperature", "atmospheric_pressure", "dTdt_filt"]]
y_train = dflr.loc[train.union(val), "qres"]
y_test = dflr.loc[test, "qres"]


reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

dflr["estimated_specific_humidity"] = dflr.loc[test, "estimated_q_outer"] - y_pred

fig = plt.figure()
plt.scatter(dflr["estimated_specific_humidity"], dflr["specific_humidity_surface"])
plt.show()

print(mae(dflr["estimated_specific_humidity"], dflr["specific_humidity_surface"]))

# %% Exporting the model
import joblib

joblib.dump(
    reg, "/Users/ea-gegan/Documents/gitrepos/air-sea-heat-flux-paper/src/bulk_variables/models/q/linear/model.pkl"
)
