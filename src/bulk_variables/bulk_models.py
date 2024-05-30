import json
import keras
from keras import layers
import numpy as np
import os
from scipy.optimize import root

base_path = os.path.dirname(os.path.realpath(__file__))


def get_train_val_test(df):
    train_idx = df.index[
        (
            (df["time"] < "2023-09-28 06:20:00+00:00")
            | ((df["time"] > "2023-10-27 23:40:00+00:00") & (df["time"] < "2023-12-01 00:00:00+00:00"))
        )
    ]
    val_idx = df.index[((df["time"] >= "2023-09-28 06:20:00+00:00") & (df["time"] <= "2023-10-27 23:40:00+00:00"))]
    test_idx = df.index[df["time"] >= "2023-12-01 00:00:00+00:00"]
    return train_idx, val_idx, test_idx


def incoming_shortwave(df):
    model = keras.models.load_model(f"{base_path}/models/shortwave")
    norms = np.load(f"{base_path}/models/shortwave/norms.npy", allow_pickle=True).item()
    df["log_battery_power"] = np.log(df["battery_power"])
    df.loc[df["battery_power"] <= 0, "log_battery_power"] = -9

    features = [
        "air_temperature",
        "solar_voltage",
        "log_battery_power",
        "sea_surface_temperature",
        "atmospheric_pressure",
        "relative_humidity",
        "U_10m_mean",
        "significant_wave_height",
    ]
    X = df[features].values
    X_norm = (X - norms["mean"]) / norms["std"]
    sw_out = model.predict(X_norm)
    return sw_out

def air_temperature_box_model(df, kappa_mdpe, length_scale):

    # Inferred shortwave required for box model estimate
    if "inferred_solar_radiation" not in df.columns:
        df["inferred_solar_radiation"] = incoming_shortwave(df)

    # Some constants
    albedo_spot = 0.1
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
    vol_mdpe_spot = (length_scale ** 3) * (4 * np.pi / 3) * (D / 2) ** 3 - vol_air_spot
    surface_area_spot = 4 * np.pi * (D / 2) ** 2

    spot_air_fraction = 0.5
    air_surface_area_spot = surface_area_spot * spot_air_fraction * (length_scale ** 2)
    water_surface_area_spot = surface_area_spot * 0.5

    # Setting up a balance
    dTdt = df["dTdt_filt"].values
    dUdt_mdpe = rho_mdpe * cp_mdpe * dTdt * vol_mdpe_spot  # kg/m^3 * J/kg K * K/s * m^3 =  W
    dUdt_air = rho_air * cp_air * dTdt * vol_air_spot  # W
    dUdt = dUdt_mdpe + dUdt_air
    R_in = df["inferred_solar_radiation"].values  # W/m^2
    T_spot_outer = (df["air_temperature"] * 0.5 + df[
        "sea_surface_temperature"] * 0.5).values  # assume the spotter shell temp is a weighted mean of the shell and water temp
    T_spot_inner = df["air_temperature"].values
    T_water = df["sea_surface_temperature"].values
    Q_solar = R_in * (1 - albedo_spot)

    # Convection to Air
    u_surface = df["U_10m_mean"].values
    Re = u_surface * 0.407 / 1.48e-5

    # Turbulent formulation, flat plate
    Nu = 0.037 * (Re ** (4 / 5))

    h_air = Nu * kappa_air / 0.407

    estimated_air_temp = np.zeros_like(T_spot_inner)
    for ii in range(len(T_spot_inner)):
        def func_zero(T_air, idx):
            return np.abs(
                dUdt[idx] - Q_solar[idx] * air_surface_area_spot
                - sigma * water_surface_area_spot * (T_water[idx] + 273.15) ** 4
                + sigma * surface_area_spot * (T_spot_outer[idx] + 273.15) ** 4
                + h_air[idx] * T_spot_outer[idx] * air_surface_area_spot
                + kappa_mdpe * air_surface_area_spot * T_spot_inner[idx] / L
                + kappa_mdpe * water_surface_area_spot * (T_spot_inner[idx] - T_water[idx]) / L
                - air_surface_area_spot * (sigma * (T_air + 273.15) ** 4 + T_air * (kappa_mdpe / L + h_air[idx]))
            )

        res = root(func_zero, x0=T_spot_outer[ii], args=(ii,))
        estimated_air_temp[ii] = res.x.item()

    return estimated_air_temp

def air_temperature_nn(df):

    def define_model(units, num_layers, activation, lr, l2):
        model_layers = [
            layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=keras.regularizers.L2(l2=l2),
                kernel_initializer=keras.initializers.HeNormal(),
            )
        ] * num_layers
        model_layers += [layers.Dense(1)]
        model = keras.Sequential(model_layers)
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=lr))

        return model

    # Making sure we have the semianalytical air temp
    if "estimated_air_temperature" not in df.columns:
        df["estimated_air_temperature"] = air_temperature_nn(df)

    with open(f"{base_path}/models/air_temp/trial.json", "r") as f:
        trial = json.load(f)
    hp = trial["hyperparameters"]["values"]
    model = define_model(
        units=hp["units"], num_layers=hp["num_layers"], activation=hp["activation"], lr=hp["lr"], l2=hp["l2"]
    )
    model.load_weights(f"{base_path}/models/air_temp/checkpoint")

    norms = np.load(f"{base_path}/models/air_temp/norms.npy", allow_pickle=True).item()

    features = [
        "air_temperature",
        "solar_voltage",
        "log_battery_power",
        "sea_surface_temperature",
        "estimated_air_temperature",
        "atmospheric_pressure",
        "relative_humidity",
        "U_10m_mean",
        "significant_wave_height",
    ]

    X = df[features].values
    X_norm = (X - norms["mean"]) / norms["std"]
    air_temp_out = model.predict(X_norm)
    return air_temp_out
