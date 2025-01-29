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
df = air_temperature_nn(df)
df["residual"] = df["estimated_air_temperature_nn"] - df["air_temperature_surface"]
df = df.loc[((df["time"] > "2024-01-15") & (df["time"] < "2024-01-25"))]
df = df.reset_index(drop=True)


#%%
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(df["air_temperature_surface"], df["sea_surface_temperature"], c=df["residual"], cmap="viridis")
cb = plt.colorbar(sc, ax=ax)
ax.set_xlabel(r"Air Temperature ($^\circ$C)")
ax.set_ylabel(r"Sea Surface Temperature ($^\circ$C)")
cb.set_label(r"$T_{\textrm{air}}$ Residual (Predicted - Actual)")
fig.tight_layout(pad=0.5)
plt.savefig("plots/tair_error.png", dpi=300)
plt.show()