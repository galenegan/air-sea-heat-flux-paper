import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import numpy as np
import pandas as pd
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


def mae(predicted, actual):
    return np.nanmean(np.abs(predicted - actual))


def rmse(predicted, actual):
    return np.sqrt(np.nanmean((predicted - actual) ** 2))


def bias(predicted, actual):
    return np.nanmean(predicted - actual)


def set_lims(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lower_lim = min(xmin, ymin)
    upper_lim = max(xmax, ymax)
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)


def get_sensible_mask(df):
    df["time"] = pd.to_datetime(df["time"], utc=True)

    dflux = np.gradient(df["sensible_heat_flux_dc"].values, df["epoch"].values)
    gradient_cutoff = 1.96 * np.nanstd(dflux)
    asit_residual = np.abs(df["sensible_heat_flux_asit_coare"].values - df["sensible_heat_flux_dc"].values)
    asit_residual_cutoff = 75
    asit_relative_residual = np.abs(df["sensible_heat_flux_dc"].values / df["sensible_heat_flux_asit_coare"].values)

    mask = (
        (df["sensible_heat_flux_dc"].values > 0)
        & (df["sensible_heat_flux_dc"].values < 300)
        & (dflux < gradient_cutoff)
        & (asit_residual < asit_residual_cutoff)
        & (asit_relative_residual < 5)
        & (df["rain_rate"].values < 1)
        & (df["epoch"].values < pd.Timestamp("2024-01-01T00:00:00Z").timestamp())
    )

    return mask


def get_latent_mask(df):
    dflux = np.gradient(df["latent_heat_flux_dc"].values, df["epoch"].values)
    gradient_cutoff = 1.96 * np.nanstd(dflux)
    asit_residual = np.abs(df["latent_heat_flux_asit_coare"].values - df["latent_heat_flux_dc"].values)
    asit_residual_cutoff = 100
    asit_relative_residual = np.abs(df["latent_heat_flux_dc"].values / df["latent_heat_flux_asit_coare"].values)

    mask = (
        (df["latent_heat_flux_dc"].values > 0)
        & (df["latent_heat_flux_dc"].values < 500)
        & (dflux < gradient_cutoff)
        & (asit_residual < asit_residual_cutoff)
        & (asit_relative_residual < 5)
        & (df["rain_rate"].values < 1)
        & (df["epoch"].values < pd.Timestamp("2024-01-01T00:00:00Z").timestamp())
    )

    return mask


# %%
project_root = get_project_root()
data_path = f"{project_root}/data"

file_dict = {
    "pure spotter": "final_flux_dataset.csv",
    "filtered": "final_flux_dataset_filtered_inputs.csv",
    "asit wind": "final_flux_dataset_asit_u.csv",
    "asit t": "final_flux_dataset_asit_t.csv",
    "asit q": "final_flux_dataset_asit_q.csv",
    "asit t and q": "final_flux_dataset_asit_q_asit_t.csv",
    "asit wind and t": "final_flux_dataset_asit_u_asit_t.csv",
    "asit wind and q": "final_flux_dataset_asit_u_asit_q.csv",
}

dfout = pd.DataFrame(index=["mae_sensible", "rmse_sensible", "bias_sensible", "mae_latent", "rmse_latent", "bias_latent"])
for col_name, file_name in file_dict.items():
    df = pd.read_csv(f"{data_path}/{file_name}")
    df = df.loc[df["spot_id"] == "31085C"]
    mask = get_sensible_mask(df)
    # Sensible heat flux
    if col_name == "filtered":
        mae_i = mae(df["sensible_heat_flux_spotter_coare"].values[mask], median_filter(df["sensible_heat_flux_dc"].values[mask], size=15))
        rmse_i = rmse(df["sensible_heat_flux_spotter_coare"].values[mask], median_filter(df["sensible_heat_flux_dc"].values[mask], size=15))
        bias_i = bias(df["sensible_heat_flux_spotter_coare"].values[mask], median_filter(df["sensible_heat_flux_dc"].values[mask], size=15))
    else:
        mae_i = mae(df["sensible_heat_flux_spotter_coare"].values[mask], df["sensible_heat_flux_dc"].values[mask])
        rmse_i = rmse(df["sensible_heat_flux_spotter_coare"].values[mask], df["sensible_heat_flux_dc"].values[mask])
        bias_i = bias(df["sensible_heat_flux_spotter_coare"].values[mask], df["sensible_heat_flux_dc"].values[mask])

    dfout.loc[["mae_sensible", "rmse_sensible", "bias_sensible"], col_name] = [mae_i, rmse_i, bias_i]

    # Latent heat flux
    mask = get_latent_mask(df)

    if col_name == "filtered":
        mae_i = mae(df["latent_heat_flux_spotter_coare"].values[mask], median_filter(df["latent_heat_flux_dc"].values[mask], size=15))
        rmse_i = rmse(df["latent_heat_flux_spotter_coare"].values[mask], median_filter(df["latent_heat_flux_dc"].values[mask], size=15))
        bias_i = bias(df["latent_heat_flux_spotter_coare"].values[mask], median_filter(df["latent_heat_flux_dc"].values[mask], size=15))
    else:
        mae_i = mae(df["latent_heat_flux_spotter_coare"].values[mask], df["latent_heat_flux_dc"].values[mask])
        rmse_i = rmse( df["latent_heat_flux_spotter_coare"].values[mask], df["latent_heat_flux_dc"].values[mask])
        bias_i = bias(df["latent_heat_flux_spotter_coare"].values[mask], df["latent_heat_flux_dc"].values[mask])

    dfout.loc[["mae_latent", "rmse_latent", "bias_latent"], col_name] = [mae_i, rmse_i, bias_i]

print(dfout.astype(float).round(2).astype(str).to_latex())
