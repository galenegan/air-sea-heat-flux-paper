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


def mae(predicted, actual):
    return np.nanmean(np.abs(predicted - actual))


def rmse(predicted, actual):
    return np.sqrt(np.nanmean((predicted - actual) ** 2))


def bias(predicted, actual):
    return np.nanmean(predicted - actual)


project_root = get_project_root()
df_full = pd.read_csv(f"{project_root}/data/bulk_variable_dataset.csv")
spot_ids_vented = ["31081C", "31084C", "31085C"]
df_full = df_full.loc[df_full["spot_id"].isin(spot_ids_vented)]
train_idx, val_idx, test_idx = get_train_val_test(df_full)

# %% Training
df = df_full.loc[train_idx].reset_index(drop=True)
indices = ["air_temperature", "shortwave", "specific_humidity"]
columns = [
    "int_mae",
    "int_rmse",
    "int_bias",
    "final_mae",
    "final_rmse",
    "final_bias",
    "delta_mae",
    "delta_rmse",
    "delta_bias",
    "pct_mae",
    "pct_rmse",
    "pct_bias",
]
out = pd.DataFrame(index=indices, columns=columns)
# Air temperature
variable = "air_temperature"
intermediate_estimate = "estimated_air_temperature"
final_estimate = "estimated_air_temperature_nn"
ground_truth_variable = "air_temperature_surface"

out.loc[variable, "int_mae"] = mae(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_rmse"] = rmse(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_bias"] = bias(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "final_mae"] = mae(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_rmse"] = rmse(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_bias"] = bias(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "delta_mae"] = out.loc[variable, "int_mae"] - out.loc[variable, "final_mae"]
out.loc[variable, "delta_rmse"] = out.loc[variable, "int_rmse"] - out.loc[variable, "final_rmse"]
out.loc[variable, "delta_bias"] = np.abs(out.loc[variable, "int_bias"]) - np.abs(out.loc[variable, "final_bias"])
out.loc[variable, "pct_mae"] = 100 * out.loc[variable, "delta_mae"] / out.loc[variable, "int_mae"]
out.loc[variable, "pct_rmse"] = 100 * out.loc[variable, "delta_rmse"] / out.loc[variable, "int_rmse"]
out.loc[variable, "pct_bias"] = 100 * out.loc[variable, "delta_bias"] / np.abs(out.loc[variable, "int_bias"])

# Shortwave
variable = "shortwave"
intermediate_estimate = "box_model_solar"
final_estimate = "estimated_shortwave_rf"
ground_truth_variable = "solar_down"

out.loc[variable, "int_mae"] = mae(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_rmse"] = rmse(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_bias"] = bias(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "final_mae"] = mae(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_rmse"] = rmse(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_bias"] = bias(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "delta_mae"] = out.loc[variable, "int_mae"] - out.loc[variable, "final_mae"]
out.loc[variable, "delta_rmse"] = out.loc[variable, "int_rmse"] - out.loc[variable, "final_rmse"]
out.loc[variable, "delta_bias"] = np.abs(out.loc[variable, "int_bias"]) - np.abs(out.loc[variable, "final_bias"])
out.loc[variable, "pct_mae"] = 100 * out.loc[variable, "delta_mae"] / out.loc[variable, "int_mae"]
out.loc[variable, "pct_rmse"] = 100 * out.loc[variable, "delta_rmse"] / out.loc[variable, "int_rmse"]
out.loc[variable, "pct_bias"] = 100 * out.loc[variable, "delta_bias"] / np.abs(out.loc[variable, "int_bias"])
# Specific Humidity
variable = "specific_humidity"
intermediate_estimate = "estimated_q_outer"
final_estimate = "estimated_specific_humidity_nn"
ground_truth_variable = "specific_humidity_surface"

out.loc[variable, "int_mae"] = mae(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_rmse"] = rmse(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_bias"] = bias(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "final_mae"] = mae(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_rmse"] = rmse(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_bias"] = bias(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "delta_mae"] = out.loc[variable, "int_mae"] - out.loc[variable, "final_mae"]
out.loc[variable, "delta_rmse"] = out.loc[variable, "int_rmse"] - out.loc[variable, "final_rmse"]
out.loc[variable, "delta_bias"] = np.abs(out.loc[variable, "int_bias"]) - np.abs(out.loc[variable, "final_bias"])
out.loc[variable, "pct_mae"] = 100 * out.loc[variable, "delta_mae"] / out.loc[variable, "int_mae"]
out.loc[variable, "pct_rmse"] = 100 * out.loc[variable, "delta_rmse"] / out.loc[variable, "int_rmse"]
out.loc[variable, "pct_bias"] = 100 * out.loc[variable, "delta_bias"] / np.abs(out.loc[variable, "int_bias"])

out = out.drop(columns=["delta_mae", "delta_rmse", "delta_bias"])
index_map = {
    "air_temperature": r"$T_{\text{air}}$",
    "shortwave": r"$Q_{\text{SW, down}}$",
    "specific_humidity": r"$q_{\text{air}}$",
}
column_map = {
    "int_mae": "Int. MAE",
    "int_rmse": "Int. RMSE",
    "int_bias": "Int. Bias",
    "final_mae": "Final MAE",
    "final_rmse": "Final RMSE",
    "final_bias": "Final Bias",
    "pct_mae": r"\% MAE",
    "pct_rmse": r"\% RMSE",
    "pct_bias": r"\% Bias",
}
out = out.rename(index=index_map, columns=column_map)
out = out.astype(float).round(decimals=2).astype(str)
print("Training data table:")
print(out.to_latex())
print(" " * 60)


# %% Validation
df = df_full.loc[val_idx].reset_index(drop=True)
indices = ["air_temperature", "shortwave", "specific_humidity"]
columns = [
    "int_mae",
    "int_rmse",
    "int_bias",
    "final_mae",
    "final_rmse",
    "final_bias",
    "delta_mae",
    "delta_rmse",
    "delta_bias",
    "pct_mae",
    "pct_rmse",
    "pct_bias",
]
out = pd.DataFrame(index=indices, columns=columns)
# Air temperature
variable = "air_temperature"
intermediate_estimate = "estimated_air_temperature"
final_estimate = "estimated_air_temperature_nn"
ground_truth_variable = "air_temperature_surface"

out.loc[variable, "int_mae"] = mae(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_rmse"] = rmse(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_bias"] = bias(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "final_mae"] = mae(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_rmse"] = rmse(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_bias"] = bias(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "delta_mae"] = out.loc[variable, "int_mae"] - out.loc[variable, "final_mae"]
out.loc[variable, "delta_rmse"] = out.loc[variable, "int_rmse"] - out.loc[variable, "final_rmse"]
out.loc[variable, "delta_bias"] = np.abs(out.loc[variable, "int_bias"]) - np.abs(out.loc[variable, "final_bias"])
out.loc[variable, "pct_mae"] = 100 * out.loc[variable, "delta_mae"] / out.loc[variable, "int_mae"]
out.loc[variable, "pct_rmse"] = 100 * out.loc[variable, "delta_rmse"] / out.loc[variable, "int_rmse"]
out.loc[variable, "pct_bias"] = 100 * out.loc[variable, "delta_bias"] / np.abs(out.loc[variable, "int_bias"])

# Shortwave
variable = "shortwave"
intermediate_estimate = "box_model_solar"
final_estimate = "estimated_shortwave_rf"
ground_truth_variable = "solar_down"

out.loc[variable, "int_mae"] = mae(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_rmse"] = rmse(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_bias"] = bias(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "final_mae"] = mae(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_rmse"] = rmse(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_bias"] = bias(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "delta_mae"] = out.loc[variable, "int_mae"] - out.loc[variable, "final_mae"]
out.loc[variable, "delta_rmse"] = out.loc[variable, "int_rmse"] - out.loc[variable, "final_rmse"]
out.loc[variable, "delta_bias"] = np.abs(out.loc[variable, "int_bias"]) - np.abs(out.loc[variable, "final_bias"])
out.loc[variable, "pct_mae"] = 100 * out.loc[variable, "delta_mae"] / out.loc[variable, "int_mae"]
out.loc[variable, "pct_rmse"] = 100 * out.loc[variable, "delta_rmse"] / out.loc[variable, "int_rmse"]
out.loc[variable, "pct_bias"] = 100 * out.loc[variable, "delta_bias"] / np.abs(out.loc[variable, "int_bias"])
# Specific Humidity
variable = "specific_humidity"
intermediate_estimate = "estimated_q_outer"
final_estimate = "estimated_specific_humidity_nn"
ground_truth_variable = "specific_humidity_surface"

out.loc[variable, "int_mae"] = mae(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_rmse"] = rmse(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_bias"] = bias(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "final_mae"] = mae(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_rmse"] = rmse(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_bias"] = bias(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "delta_mae"] = out.loc[variable, "int_mae"] - out.loc[variable, "final_mae"]
out.loc[variable, "delta_rmse"] = out.loc[variable, "int_rmse"] - out.loc[variable, "final_rmse"]
out.loc[variable, "delta_bias"] = np.abs(out.loc[variable, "int_bias"]) - np.abs(out.loc[variable, "final_bias"])
out.loc[variable, "pct_mae"] = 100 * out.loc[variable, "delta_mae"] / out.loc[variable, "int_mae"]
out.loc[variable, "pct_rmse"] = 100 * out.loc[variable, "delta_rmse"] / out.loc[variable, "int_rmse"]
out.loc[variable, "pct_bias"] = 100 * out.loc[variable, "delta_bias"] / np.abs(out.loc[variable, "int_bias"])

out = out.drop(columns=["delta_mae", "delta_rmse", "delta_bias"])
index_map = {
    "air_temperature": r"$T_{\text{air}}$",
    "shortwave": r"$Q_{\text{SW, down}}$",
    "specific_humidity": r"$q_{\text{air}}$",
}
column_map = {
    "int_mae": "Int. MAE",
    "int_rmse": "Int. RMSE",
    "int_bias": "Int. Bias",
    "final_mae": "Final MAE",
    "final_rmse": "Final RMSE",
    "final_bias": "Final Bias",
    "pct_mae": r"\% MAE",
    "pct_rmse": r"\% RMSE",
    "pct_bias": r"\% Bias",
}
out = out.rename(index=index_map, columns=column_map)
out = out.astype(float).round(decimals=2).astype(str)
print("Validation data table:")
print(out.to_latex())
print(" " * 60)
# %% Test
df = df_full.loc[test_idx].reset_index(drop=True)
indices = ["air_temperature", "shortwave", "specific_humidity"]
columns = [
    "int_mae",
    "int_rmse",
    "int_bias",
    "final_mae",
    "final_rmse",
    "final_bias",
    "delta_mae",
    "delta_rmse",
    "delta_bias",
    "pct_mae",
    "pct_rmse",
    "pct_bias",
]
out = pd.DataFrame(index=indices, columns=columns)
# Air temperature
variable = "air_temperature"
intermediate_estimate = "estimated_air_temperature"
final_estimate = "estimated_air_temperature_nn"
ground_truth_variable = "air_temperature_surface"

out.loc[variable, "int_mae"] = mae(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_rmse"] = rmse(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_bias"] = bias(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "final_mae"] = mae(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_rmse"] = rmse(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_bias"] = bias(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "delta_mae"] = out.loc[variable, "int_mae"] - out.loc[variable, "final_mae"]
out.loc[variable, "delta_rmse"] = out.loc[variable, "int_rmse"] - out.loc[variable, "final_rmse"]
out.loc[variable, "delta_bias"] = np.abs(out.loc[variable, "int_bias"]) - np.abs(out.loc[variable, "final_bias"])
out.loc[variable, "pct_mae"] = 100 * out.loc[variable, "delta_mae"] / out.loc[variable, "int_mae"]
out.loc[variable, "pct_rmse"] = 100 * out.loc[variable, "delta_rmse"] / out.loc[variable, "int_rmse"]
out.loc[variable, "pct_bias"] = 100 * out.loc[variable, "delta_bias"] / np.abs(out.loc[variable, "int_bias"])

# Shortwave
variable = "shortwave"
intermediate_estimate = "box_model_solar"
final_estimate = "estimated_shortwave_rf"
ground_truth_variable = "solar_down"

out.loc[variable, "int_mae"] = mae(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_rmse"] = rmse(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_bias"] = bias(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "final_mae"] = mae(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_rmse"] = rmse(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_bias"] = bias(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "delta_mae"] = out.loc[variable, "int_mae"] - out.loc[variable, "final_mae"]
out.loc[variable, "delta_rmse"] = out.loc[variable, "int_rmse"] - out.loc[variable, "final_rmse"]
out.loc[variable, "delta_bias"] = np.abs(out.loc[variable, "int_bias"]) - np.abs(out.loc[variable, "final_bias"])
out.loc[variable, "pct_mae"] = 100 * out.loc[variable, "delta_mae"] / out.loc[variable, "int_mae"]
out.loc[variable, "pct_rmse"] = 100 * out.loc[variable, "delta_rmse"] / out.loc[variable, "int_rmse"]
out.loc[variable, "pct_bias"] = 100 * out.loc[variable, "delta_bias"] / np.abs(out.loc[variable, "int_bias"])
# Specific Humidity
variable = "specific_humidity"
intermediate_estimate = "estimated_q_outer"
final_estimate = "estimated_specific_humidity_nn"
ground_truth_variable = "specific_humidity_surface"

out.loc[variable, "int_mae"] = mae(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_rmse"] = rmse(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "int_bias"] = bias(df[intermediate_estimate], df[ground_truth_variable])
out.loc[variable, "final_mae"] = mae(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_rmse"] = rmse(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "final_bias"] = bias(df[final_estimate], df[ground_truth_variable])
out.loc[variable, "delta_mae"] = out.loc[variable, "int_mae"] - out.loc[variable, "final_mae"]
out.loc[variable, "delta_rmse"] = out.loc[variable, "int_rmse"] - out.loc[variable, "final_rmse"]
out.loc[variable, "delta_bias"] = np.abs(out.loc[variable, "int_bias"]) - np.abs(out.loc[variable, "final_bias"])
out.loc[variable, "pct_mae"] = 100 * out.loc[variable, "delta_mae"] / out.loc[variable, "int_mae"]
out.loc[variable, "pct_rmse"] = 100 * out.loc[variable, "delta_rmse"] / out.loc[variable, "int_rmse"]
out.loc[variable, "pct_bias"] = 100 * out.loc[variable, "delta_bias"] / np.abs(out.loc[variable, "int_bias"])

out = out.drop(columns=["delta_mae", "delta_rmse", "delta_bias"])
index_map = {
    "air_temperature": r"$T_{\text{air}}$",
    "shortwave": r"$Q_{\text{SW, down}}$",
    "specific_humidity": r"$q_{\text{air}}$",
}
column_map = {
    "int_mae": "Int. MAE",
    "int_rmse": "Int. RMSE",
    "int_bias": "Int. Bias",
    "final_mae": "Final MAE",
    "final_rmse": "Final RMSE",
    "final_bias": "Final Bias",
    "pct_mae": r"\% MAE",
    "pct_rmse": r"\% RMSE",
    "pct_bias": r"\% Bias",
}
out = out.rename(index=index_map, columns=column_map)
out = out.astype(float).round(decimals=2).astype(str)
print("Test data table:")
print(out.to_latex())
print(" " * 60)
