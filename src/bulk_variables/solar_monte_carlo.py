"""
Generates Figure 9
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import get_project_root
from src.bulk_variables.bulk_models import air_temperature_linear, get_train_val_test

np.random.seed(41)

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
train_idx, val_idx, test_idx = get_train_val_test(df)
df = df.loc[test_idx]
df = df.reset_index(drop=True)
df = air_temperature_linear(df)


def incoming_shortwave_box_model(df: pd.DataFrame) -> np.array:
    """
    Add the box model shortwave radiation estimate to a dataframe
    """

    x = np.load(f"{project_root}/src/bulk_variables/models/shortwave/box/optimal_params.npy", allow_pickle=True)
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
        df["air_temperature"] * weight_ti + df["air_temp_to_use"] * weight_ta
    )  # assume the spotter shell temp is a weighted mean of the internal and external air temps
    T_spot_inner = df["air_temperature"]
    T_water = df["sea_surface_temperature"]
    T_air = df["air_temp_to_use"]

    # Convection to Air
    u_surface = df["U_10_to_use"]
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

    return estimated_solar


# %% Error metrics
rmse_tair = 1.14
bias_tair = 0.09
random_error_tair = np.random.normal(loc=bias_tair, scale=rmse_tair, size=len(df))

rmse_wind = 1.931
bias_wind = 0.1366
random_error_wind = np.random.normal(loc=bias_wind, scale=rmse_wind, size=len(df))

# %% Running standard case
df["air_temp_to_use"] = df["estimated_air_temperature"]
df["U_10_to_use"] = df["U_10m_mean"]
df["box_model_solar"] = incoming_shortwave_box_model(df)
df["residual"] = df["box_model_solar"] - df["solar_down"]

# Temperature perturbed case
df["air_temp_to_use"] = df["estimated_air_temperature"] + random_error_tair
df["box_model_solar_temp_perturbed"] = incoming_shortwave_box_model(df)
df["residual_temp_perturbed"] = df["box_model_solar_temp_perturbed"] - df["solar_down"]

# Wind perturbed case
df["air_temp_to_use"] = df["estimated_air_temperature"]
df["U_10_to_use"] = df["U_10m_mean"] + random_error_wind
df["box_model_solar_wind_perturbed"] = incoming_shortwave_box_model(df)
df["residual_wind_perturbed"] = df["box_model_solar_wind_perturbed"] - df["solar_down"]

# Both perturbed case
df["air_temp_to_use"] = df["estimated_air_temperature"] + random_error_tair
# df["U_10_to_use"] = df["U_10m_mean"] + random_error_wind
df["box_model_solar_both_perturbed"] = incoming_shortwave_box_model(df)
df["residual_both_perturbed"] = df["box_model_solar_both_perturbed"] - df["solar_down"]

# %% Plotting
bins = np.arange(-750, 751, 5)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

bias = df["residual"].mean()
mae = np.nanmean(df["residual"].abs())
rmse = np.nanstd(df["residual"])
ax1.hist(df["residual"], bins=bins, color="#012749", density=True)
ax1.set_title("(a)")
ax1.set_ylabel("Density")
ax1.set_xlabel(r"Shortwave Residual (W/m$^2$)")
ax1.set_yscale("log")
ax1.annotate(
    f"MAE = {mae:.2f} " + r"W/m$^2$" + f"\nRMSE = {rmse:.2f} " + r"W/m$^2$" + f"\nBias = {bias:.2f} " + r"W/m$^2$",
    xy=(0.01, 0.8),
    xycoords="axes fraction",
)
ax1.set_xlim(-750, 750)

bias = df["residual_temp_perturbed"].mean()
mae = np.nanmean(df["residual_temp_perturbed"].abs())
rmse = np.nanstd(df["residual_temp_perturbed"])
ax2.hist(df["residual_temp_perturbed"], bins=bins, color="#012749", density=True)
ax2.set_title("(b)")
ax2.set_xlabel(r"Shortwave Residual with Temp Perturbed (W/m$^2$)")
ax2.set_ylabel("Density")
ax2.annotate(
    f"MAE = {mae:.2f} " + r"W/m$^2$" + f"\nRMSE = {rmse:.2f} " + r"W/m$^2$" + f"\nBias = {bias:.2f} " + r"W/m$^2$",
    xy=(0.01, 0.8),
    xycoords="axes fraction",
)
ax2.set_yscale("log")
ax2.set_xlim(-750, 750)

bias = df["residual_wind_perturbed"].mean()
mae = np.nanmean(df["residual_wind_perturbed"].abs())
rmse = np.nanstd(df["residual_wind_perturbed"])
ax3.hist(df["residual_wind_perturbed"], bins=bins, color="#012749", density=True)
ax3.set_title("(c)")
ax3.set_xlabel(r"Shortwave Residual with Wind Perturbed (W/m$^2$)")
ax3.set_ylabel("Density")
ax3.annotate(
    f"MAE = {mae:.2f} " + r"W/m$^2$" + f"\nRMSE = {rmse:.2f} " + r"W/m$^2$" + f"\nBias = {bias:.2f} " + r"W/m$^2$",
    xy=(0.01, 0.8),
    xycoords="axes fraction",
)
ax3.set_xlim(-750, 750)
ax3.set_yscale("log")

bias = df["residual_both_perturbed"].mean()
mae = np.nanmean(df["residual_both_perturbed"].abs())
rmse = np.nanstd(df["residual_both_perturbed"])
ax4.hist(df["residual_both_perturbed"], bins=bins, color="#012749", density=True)
ax4.set_title("(d)")
ax4.set_xlabel(r"Shortwave Residual with Both Perturbed (W/m$^2$)")
ax4.set_ylabel("Density")
ax4.annotate(
    f"MAE = {mae:.2f} " + r"W/m$^2$" + f"\nRMSE = {rmse:.2f} " + r"W/m$^2$" + f"\nBias = {bias:.2f} " + r"W/m$^2$",
    xy=(0.01, 0.8),
    xycoords="axes fraction",
)
ax4.set_yscale("log")
ax4.set_xlim(-750, 750)
fig.tight_layout(pad=0.5)
# plt.savefig(f"{project_root}/plots/solar_mc.png", dpi=300)
plt.show()
