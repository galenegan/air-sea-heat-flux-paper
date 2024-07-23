import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
from bulk_models import air_temperature_box_model, get_train_val_test, incoming_shortwave
from src.utils import get_project_root

# Data and i/o
file_path = os.path.dirname(os.path.realpath(__file__))
project_root = get_project_root()
spot_ids_vented = ["31081C", "31084C", "31085C"]
data_path = f"{project_root}/data"
df_full = pd.read_csv(f"{data_path}/training_dataset.csv")
df_vented = df_full.loc[df_full["spot_id"].isin(spot_ids_vented)]
train_idx, val_idx, test_idx = get_train_val_test(df_vented)
df = df_vented.loc[train_idx.union(val_idx)]
df = df.sort_values(by="epoch").reset_index(drop=True)
df["inferred_solar_radiation"] = incoming_shortwave(df)


def cost_function(x, df):
    # Unknowns
    kappa_mdpe, length_scale = x[0], x[1]
    df["estimated_air_temperature"] = air_temperature_box_model(df, kappa_mdpe, length_scale)
    error = df["air_temperature_surface"] - df["estimated_air_temperature"]
    cost = np.nanmean(error**2)
    return cost


res = minimize(cost_function, x0=[0.28, 1.0], bounds=((0.05, 0.5), (1.0, 1.4)), args=(df,))
optimal_params = {"kappa": res.x[0], "length_scale": res.x[1]}
np.save(f"{file_path}/models/air_temp/box_model_params_unvented.npy", optimal_params)
