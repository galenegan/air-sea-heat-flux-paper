import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import numpy as np
import pandas as pd

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


def mae(x, y):
    return np.nanmean(np.abs(x - y))


def bias(x, y):
    return np.nanmean(x - y)


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
    spot_asit_residual = np.abs(
        df["sensible_heat_flux_asit_coare"].values - df["sensible_heat_flux_spotter_coare"].values
    )
    asit_residual_cutoff = 50
    residual_gradient = np.gradient(asit_residual, df["epoch"].values)
    asit_relative_residual = np.abs(df["sensible_heat_flux_dc"].values / df["sensible_heat_flux_asit_coare"].values)

    mask = (
        (df["sensible_heat_flux_dc"].values > -300)
        & (df["sensible_heat_flux_dc"].values < 300)
        & (dflux < gradient_cutoff)
        & (asit_residual < asit_residual_cutoff)
        & (asit_relative_residual < 5)
        & (df["rain_rate_13m"].values < 1)
        & (df["epoch"].values >= pd.Timestamp("2023-12-01T00:00:00Z").timestamp())
        & (df["epoch"].values <= pd.Timestamp("2024-01-10T00:00:00Z").timestamp())
    )

    return mask


def get_latent_mask(df):
    dflux = np.gradient(df["latent_heat_flux_dc"].values, df["epoch"].values)
    gradient_cutoff = np.nanstd(dflux)
    asit_residual = np.abs(df["latent_heat_flux_asit_coare"].values - df["latent_heat_flux_dc"].values)
    spot_asit_residual = np.abs(df["latent_heat_flux_asit_coare"].values - df["latent_heat_flux_spotter_coare"].values)
    asit_residual_cutoff = 75
    asit_relative_residual = np.abs(df["latent_heat_flux_dc"].values / df["latent_heat_flux_asit_coare"].values)

    mask = (
        (df["latent_heat_flux_dc"].values > -300)
        & (df["latent_heat_flux_dc"].values < 500)
        & (dflux < gradient_cutoff)
        & (asit_residual < asit_residual_cutoff)
        & (asit_relative_residual < 5)
        & (df["rain_rate_13m"].values < 1)
        & (df["epoch"].values >= pd.Timestamp("2023-12-01T00:00:00Z").timestamp())
        & (df["epoch"].values <= pd.Timestamp("2024-01-10T00:00:00Z").timestamp())
    )

    return mask


# Train test split by time

# %%
data_path = "../../data"

file_dict = {
    "pure spotter": "final_dataset.csv",
    "asit wind": "final_dataset_asit_u.csv",
    "asit t": "final_dataset_asit_t.csv",
    "asit q": "final_dataset_asit_q.csv",
    "asit t and q": "final_dataset_asit_q_asit_t.csv",
    "asit wind and t": "final_dataset_asit_u_asit_t.csv",
    "asit wind and q": "final_dataset_asit_u_asit_q.csv",
}

dfout = pd.DataFrame(index=["mae_sensible", "bias_sensible", "mae_latent", "bias_latent"])
for col_name, file_name in file_dict.items():
    df = pd.read_csv(f"{data_path}/{file_name}")

    # Sensible heat flux
    mask = get_sensible_mask(df)
    mae_i = mae(df["sensible_heat_flux_dc"].values[mask], df["sensible_heat_flux_spotter_coare"].values[mask])
    bias_i = bias(df["sensible_heat_flux_spotter_coare"].values[mask], df["sensible_heat_flux_dc"].values[mask])
    dfout.loc[["mae_sensible", "bias_sensible"], col_name] = [mae_i, bias_i]

    if file_name == "final_dataset.csv":
        # Size 15 (5 hours) does best:
        sensible_filtered = median_filter(df["sensible_heat_flux_spotter_coare"].values, size=15)
        mae_sens = mae(df["sensible_heat_flux_dc"].values[mask], sensible_filtered[mask])
        bias_sens = bias(sensible_filtered[mask], df["sensible_heat_flux_dc"].values[mask])
        dfout.loc[["mae_sensible", "bias_sensible"], "median filtered"] = [mae_sens, bias_sens]

    # Latent heat flux
    mask = get_latent_mask(df)
    mae_i = mae(df["latent_heat_flux_dc"].values[mask], df["latent_heat_flux_spotter_coare"].values[mask])
    bias_i = bias(df["latent_heat_flux_spotter_coare"].values[mask], df["latent_heat_flux_dc"].values[mask])
    dfout.loc[["mae_latent", "bias_latent"], col_name] = [mae_i, bias_i]

    if file_name == "final_dataset.csv":
        # Size 15 (5 hours) does best:
        latent_filtered = median_filter(df["latent_heat_flux_spotter_coare"].values, size=15)
        mae_latent = mae(df["latent_heat_flux_dc"].values[mask], latent_filtered[mask])
        bias_latent = bias(latent_filtered[mask], df["latent_heat_flux_dc"].values[mask])
        dfout.loc[["mae_latent", "bias_latent"], "median filtered"] = [mae_latent, bias_latent]


print(dfout.astype(float).round(2).astype(str).to_latex())
