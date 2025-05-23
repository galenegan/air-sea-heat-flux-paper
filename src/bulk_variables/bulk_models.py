import joblib
import json
import keras
from keras import layers, regularizers
import numpy as np
import os
import pandas as pd
from typing import Tuple
from src.coare3p5.meteo import qair

base_path = os.path.dirname(os.path.realpath(__file__))


def get_train_val_test(df: pd.DataFrame) -> Tuple:
    """
    Training, validation, and test split
    """
    train_idx = df.index[df["time"] >= "2024-02-01"]
    val_idx = df.index[((df["time"] >= "2024-01-01") & (df["time"] < "2024-02-01"))]
    test_idx = df.index[df["time"] < "2024-01-01"]
    return train_idx, val_idx, test_idx


def air_temperature_linear(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the linear air temperature estimate and related variables to a dataframe
    """

    def fill_night_temp(df):
        dfout = pd.DataFrame()
        for spot_id in df["spot_id"].unique():
            dfs = df.loc[df["spot_id"] == spot_id]
            dfs["night_temp"] = dfs.loc[dfs["solar_voltage"] < 1, "air_temperature"]
            dfs["night_sst"] = dfs.loc[dfs["solar_voltage"] < 1, "sea_surface_temperature"]
            dfs["night_temp_interp"] = dfs["night_temp"].interpolate(method="ffill")
            dfs["night_sst_interp"] = dfs["night_sst"].interpolate(method="ffill")
            dfs["delta_night_air_temp"] = dfs["air_temperature"] - dfs["night_temp_interp"]
            dfs["delta_night_sst"] = dfs["sea_surface_temperature"] - dfs["night_sst_interp"]
            dfout = pd.concat([dfout, dfs])
        return dfout

    df = fill_night_temp(df)
    df = df.sort_index()
    X = np.vstack((df["delta_night_air_temp"].values, df["delta_night_sst"].values)).T
    pred_idx = ~np.isnan(X).any(axis=1)

    # Loading the linear model
    model = joblib.load(f"{base_path}/models/air_temp/linear/model.pkl")
    estimated_air_temperature = np.zeros((X.shape[0],)) * np.nan
    estimated_air_temperature[pred_idx] = (
        df["air_temperature"].values[pred_idx] - model.predict(X[pred_idx, :]).squeeze()
    )
    df["estimated_air_temperature"] = estimated_air_temperature
    return df


def air_temperature_nn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the neural network air temperature estimate to a dataframe
    """

    def define_model(units, num_layers, activation, l2):
        model_layers = [
            layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.L2(l2=l2),
                kernel_initializer=keras.initializers.HeNormal(),
            )
        ] * num_layers
        model_layers += [layers.Dense(1)]
        model = keras.Sequential(model_layers)
        model.compile()

        return model

    # Making sure we have the semi-analytical air temp
    if "estimated_air_temperature" not in df.columns:
        df = air_temperature_linear(df)

    with open(f"{base_path}/models/air_temp/neural_net/trial.json", "r") as f:
        trial = json.load(f)
    hp = trial["hyperparameters"]["values"]
    model = define_model(units=hp["units"], num_layers=hp["num_layers"], activation=hp["activation"], l2=hp["l2"])
    model.load_weights(f"{base_path}/models/air_temp/neural_net/checkpoint")

    norms = np.load(f"{base_path}/models/air_temp/neural_net/norms.npy", allow_pickle=True).item()

    features = [
        "estimated_air_temperature",
        "air_temperature",
        "solar_voltage",
        "sea_surface_temperature",
        "U_10m_mean",
    ]

    X = df[features].values
    X_norm = (X - norms["mean"]) / norms["std"]

    # Neural net predicts the residual
    bad_indices = np.any(pd.isna(df[features + ["air_temperature_surface"]]), axis=1)
    df["estimated_air_temperature_nn"] = df["estimated_air_temperature"] - model.predict(X_norm).squeeze()
    df.loc[bad_indices, "estimated_air_temperature_nn"] = np.nan
    return df


def incoming_shortwave_box_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the box model shortwave radiation estimate to a dataframe
    """

    if "estimated_air_temperature" not in df.columns:
        df = air_temperature_linear(df)

    x = np.load(f"{base_path}/models/shortwave/box/optimal_params_revised.npy", allow_pickle=True)
    kappa_mdpe, length_scale, albedo_spot, weight_ta, weight_ti = x[0], x[1], x[2], x[3], x[4]

    # Some constants
    cp_air = 1012
    cp_mdpe = 2300
    rho_air = 1.225
    rho_mdpe = 939
    sigma = 5.670e-8  # Boltzmann constant
    kappa_air = 0.025

    # Approximate because the real geometry is complicated
    D = 0.297  # spotter diameter, m
    L = 0.01  # spotter thickness, m
    vol_air_spot = (4 * np.pi / 3) * ((D - L) / 2) ** 3
    vol_mdpe_spot = (length_scale**3) * (4 * np.pi / 3) * (D / 2) ** 3 - vol_air_spot
    surface_area_spot = 4 * np.pi * (D / 2) ** 2

    spot_air_fraction = 0.5
    air_surface_area_spot = surface_area_spot * spot_air_fraction * (length_scale**2)
    water_surface_area_spot = surface_area_spot * 0.5

    # Unsteadiness
    dTdt = df["dTdt_filt"].values

    # Setting up a balance
    dUdt_mdpe = rho_mdpe * cp_mdpe * dTdt * vol_mdpe_spot  # kg/m^3 * J/kg K * K/s * m^3 =  W
    dUdt_air = rho_air * cp_air * dTdt * vol_air_spot  # W
    dUdt = dUdt_mdpe + dUdt_air
    T_spot_outer = (
        df["air_temperature"] * weight_ti + df["estimated_air_temperature"] * weight_ta
    )  # assume the spotter shell temp is a weighted mean of the internal and external air temps
    T_spot_inner = df["air_temperature"]
    T_water = df["sea_surface_temperature"]
    T_air = df["estimated_air_temperature"]

    # Convection to Air
    u_surface = df["U_10m_mean"]
    Re = u_surface * 0.407 / 1.48e-5

    # Turbulent formulation, flat plate
    Nu = 0.0296 * (Re ** (4 / 5))
    h_air = Nu * kappa_air / 0.407

    estimated_solar = (
        (
            dUdt
            - sigma * water_surface_area_spot * (T_water + 273.15) ** 4
            + sigma * surface_area_spot * (T_spot_outer + 273.15) ** 4
            + h_air * T_spot_outer * air_surface_area_spot
            + kappa_mdpe * air_surface_area_spot * T_spot_inner / L
            + kappa_mdpe * water_surface_area_spot * (T_spot_inner - T_water) / L
            - air_surface_area_spot * (sigma * (T_air + 273.15) ** 4 + T_air * (kappa_mdpe / L + h_air))
        )
        / air_surface_area_spot
        / (1 - albedo_spot)
    )
    estimated_solar = np.maximum(estimated_solar, 0)
    df["box_model_solar"] = estimated_solar
    return df


def incoming_shortwave_random_forest(df: pd.DataFrame, version="revised") -> pd.DataFrame:
    """
    Add the random forest shortwave estimate to a dataframe
    """

    if "box_model_solar" not in df.columns:
        df = incoming_shortwave_box_model(df)

    model = joblib.load(f"{base_path}/models/shortwave/random_forest/{version}/model.pkl")
    norms = np.load(f"{base_path}/models/shortwave/random_forest/{version}/norms.npy", allow_pickle=True).item()

    if version == "original":
        features = [
            "box_model_solar",
            "delta_night_air_temp",
            "solar_voltage",
            "log_battery_power",
            "sea_surface_temperature",
        ]
        X = df.loc[:, features].values
        X_norm = (X - norms["mean"]) / norms["std"]
        shortwave_out = df["box_model_solar"].values - model.predict(X_norm).squeeze()
        df["estimated_shortwave_rf"] = np.maximum(shortwave_out, 0)
    elif version == "revised":
        features = [
            "box_model_solar",
            "delta_night_air_temp",
            "estimated_air_temperature",
            "solar_voltage",
            "battery_power",
        ]
        X = df.loc[:, features]
        X_norm = (X - norms["mean"]) / norms["std"]
        shortwave_out = model.predict(X_norm).squeeze()
        bad_indices = np.any(pd.isna(df[features + ["solar_down"]]), axis=1)
        df["estimated_shortwave_rf"] = shortwave_out
        df.loc[bad_indices, "estimated_shortwave_rf"] = np.nan

    return df


def specific_humidity_exp_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the exponential decay humidity estimate to a dataframe
    """
    x = np.load(f"{base_path}/models/q/exponential_decay/optimal_params.npy", allow_pickle=True)
    a, b, c = x[0], x[1], x[2]

    df["q_inner"], _ = qair(df["air_temperature"], df["atmospheric_pressure"], df["relative_humidity"])
    df["days"] = (df["epoch"] - df["epoch"].min()) / 86400
    estimated_q_outer = df["q_inner"].values + (a * np.exp(b * df["days"].values) + c)
    df["estimated_q_outer"] = estimated_q_outer
    return df


def specific_humidity_linear(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the linear specific humidity estimate and related variables to a dataframe
    """
    if "estimated_air_temperature" not in df.columns:
        df = air_temperature_linear(df)

    if "estimated_q_outer" not in df.columns:
        df = specific_humidity_exp_decay(df)

    df = df.sort_index()
    X = np.vstack((df["estimated_air_temperature"].values, df["atmospheric_pressure"].values, df["dTdt_filt"].values)).T
    pred_idx = ~np.isnan(X).any(axis=1)

    # Loading the linear model
    model = joblib.load(f"{base_path}/models/q/linear/model.pkl")
    estimated_specific_humidity = np.zeros((X.shape[0],)) * np.nan
    estimated_specific_humidity[pred_idx] = (
        df["estimated_q_outer"].values[pred_idx] - model.predict(X[pred_idx, :]).squeeze()
    )
    df["estimated_specific_humidity"] = estimated_specific_humidity
    return df


def specific_humidity_nn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the neural network humidity estimate to a dataframe
    """

    def define_model(units, num_layers, activation, l2):
        model_layers = [
            layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.L2(l2=l2),
                kernel_initializer=keras.initializers.HeNormal(),
            )
        ] * num_layers
        model_layers += [layers.Dense(1)]
        model = keras.Sequential(model_layers)
        model.compile()

        return model

    # Making sure we have the rough estimates of air temp and q
    if "estimated_q_outer" not in df.columns:
        df = specific_humidity_exp_decay(df)

    if "estimated_air_temperature" not in df.columns:
        df = air_temperature_linear(df)

    with open(f"{base_path}/models/q/neural_net/trial.json", "r") as f:
        trial = json.load(f)
    hp = trial["hyperparameters"]["values"]
    model = define_model(units=hp["units"], num_layers=hp["num_layers"], activation=hp["activation"], l2=hp["l2"])
    model.load_weights(f"{base_path}/models/q/neural_net/checkpoint")

    norms = np.load(f"{base_path}/models/q/neural_net/norms.npy", allow_pickle=True).item()

    features = [
        "estimated_air_temperature",
        "estimated_q_outer",
        "solar_voltage",
        "sea_surface_temperature",
        "U_10m_mean",
    ]

    X = df.loc[:, features].values
    X_norm = (X - norms["mean"]) / norms["std"]
    q_out = df["estimated_q_outer"].values - model.predict(X_norm).squeeze()
    bad_indices = np.any(pd.isna(df[features + ["specific_humidity_surface"]]), axis=1)
    df["estimated_specific_humidity_nn"] = q_out
    df.loc[bad_indices, "estimated_specific_humidity_nn"] = np.nan
    return df
