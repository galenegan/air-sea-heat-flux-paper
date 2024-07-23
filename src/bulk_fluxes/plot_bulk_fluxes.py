import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
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


project_root = get_project_root()
data_path = f"{project_root}/data"
df = pd.read_csv(f"{data_path}/final_flux_dataset.csv")
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.loc[df["spot_id"] == "31085C"]  # Last remaining vented Spotter during test set

# %%
dflux = np.gradient(df["sensible_heat_flux_dc"].values, df["epoch"].values)
gradient_cutoff = 1.96 * np.nanstd(dflux)
asit_residual = np.abs(df["sensible_heat_flux_asit_coare"].values - df["sensible_heat_flux_dc"].values)
spot_asit_residual = np.abs(df["sensible_heat_flux_asit_coare"].values - df["sensible_heat_flux_spotter_coare"].values)
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
)


# %% Setting up the plot
fig = plt.figure(constrained_layout=True)
gs = GridSpec(4, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])
ax6 = fig.add_subplot(gs[3, :])

one = np.linspace(
    np.nanmin(df["sensible_heat_flux_dc"].values[mask]), np.nanmax(df["sensible_heat_flux_dc"].values[mask]), 100
)
ax1.plot(
    df["sensible_heat_flux_asit_coare"].values[mask],
    df["sensible_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    alpha=1.0,
    markersize=5,
)
ax1.plot(
    one,
    one,
    "-",
    color="#9f1853",
    linewidth=3,
    label=(
        f"MAE = {mae(df['sensible_heat_flux_dc'][mask], df['sensible_heat_flux_asit_coare'][mask]):.2f} W/m$^2$\n"
        + f"Bias = {np.nanmean(df['sensible_heat_flux_asit_coare'][mask] - df['sensible_heat_flux_dc'][mask]):.2f} W/m$^2$"
    ),
)
ax1.set_xlabel("ASIT Bulk Sensible (W/m$^2$)")
ax1.set_ylabel("Direct Covariance Sensible (W/m$^2$)")
ax1.legend(loc="upper left", fontsize=12)
ax1.set_title("(a)")
set_lims(ax1)

ax2.plot(
    df["sensible_heat_flux_spotter_coare"].values[mask],
    df["sensible_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    alpha=1.0,
    markersize=5,
)
ax2.plot(
    one,
    one,
    "-",
    color="#9f1853",
    linewidth=3,
    label=(
        f"MAE = {mae(df['sensible_heat_flux_dc'][mask], df['sensible_heat_flux_spotter_coare'][mask]):.2f} W/m$^2$\n"
        + f"Bias = {np.nanmean(df['sensible_heat_flux_spotter_coare'][mask] - df['sensible_heat_flux_dc'][mask]):.2f} W/m$^2$"
    ),
)
ax2.set_xlabel("Spotter Bulk Sensible (W/m$^2$)")
ax2.set_ylabel("Direct Covariance Sensible (W/m$^2$)")
ax2.legend(loc="upper left", fontsize=12)
ax2.set_title("(b)")
set_lims(ax2)

ax3.plot(
    df["time"].values[mask],
    df["sensible_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    markersize=4,
    label="Direct Covariance",
)
ax3.plot(
    df["time"].values[mask],
    df["sensible_heat_flux_asit_coare"].values[mask],
    color="#6929c4",
    alpha=0.8,
    linewidth=2.5,
    label="ASIT Bulk",
)
ax3.plot(
    df["time"].values[mask],
    df["sensible_heat_flux_spotter_coare"].values[mask],
    "-",
    color="#009d9a",
    alpha=0.8,
    linewidth=2,
    label="Spotter Bulk",
)
ax3.set_ylabel(r"Sensible Heat Flux (W/m$^2$)")
ax3.legend()
plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax3.set_title("(c)")

# %% Now getting the latent heat flux
dflux = np.gradient(df["latent_heat_flux_dc"].values, df["epoch"].values)
gradient_cutoff = np.nanstd(dflux)
asit_residual = np.abs(df["latent_heat_flux_asit_coare"].values - df["latent_heat_flux_dc"].values)
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
)
one = np.linspace(
    np.nanmin(df["latent_heat_flux_dc"].values[mask]), np.nanmax(df["latent_heat_flux_dc"].values[mask]), 100
)

ax4.plot(
    df["latent_heat_flux_asit_coare"].values[mask],
    df["latent_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    alpha=1.0,
    markersize=5,
)
ax4.plot(
    one,
    one,
    "-",
    color="#9f1853",
    linewidth=3,
    label=(
        f"MAE = {mae(df['latent_heat_flux_dc'][mask], df['latent_heat_flux_asit_coare'][mask]):.2f} W/m$^2$\n"
        + f"Bias = {np.nanmean(df['latent_heat_flux_asit_coare'][mask] - df['latent_heat_flux_dc'][mask]):.2f} W/m$^2$"
    ),
)
ax4.set_xlabel("ASIT Bulk Latent (W/m$^2$)")
ax4.set_ylabel("Direct Covariance Latent (W/m$^2$)")
ax4.legend(loc="upper left", fontsize=12)
ax4.set_title("(d)")
set_lims(ax4)

ax5.plot(
    df["latent_heat_flux_spotter_coare"].values[mask],
    df["latent_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    alpha=1.0,
    markersize=5,
)
ax5.plot(
    one,
    one,
    "-",
    color="#9f1853",
    linewidth=3,
    label=(
        f"MAE = {mae(df['latent_heat_flux_dc'][mask], df['latent_heat_flux_spotter_coare'][mask]):.2f} W/m$^2$\n"
        + f"Bias = {np.nanmean(df['latent_heat_flux_spotter_coare'][mask] - df['latent_heat_flux_dc'][mask]):.2f} W/m$^2$"
    ),
)
ax5.set_xlabel("Spotter Bulk Latent (W/m$^2$)")
ax5.set_ylabel("Direct Covariance Latent (W/m$^2$)")
ax5.legend(loc="upper left", fontsize=12)
ax5.set_title("(e)")
set_lims(ax5)

ax6.plot(
    df["time"].values[mask],
    df["latent_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    markersize=4,
    label="Direct Covariance",
)
ax6.plot(
    df["time"].values[mask],
    df["latent_heat_flux_asit_coare"].values[mask],
    color="#6929c4",
    alpha=0.8,
    linewidth=2.5,
    label="ASIT Bulk",
)
ax6.plot(
    df["time"].values[mask],
    df["latent_heat_flux_spotter_coare"].values[mask],
    "-",
    color="#009d9a",
    alpha=0.8,
    linewidth=2,
    label="Spotter Bulk",
)
ax6.set_ylabel(r"Latent Heat Flux (W/m$^2$)")
ax6.legend(loc="lower right")
plt.setp(ax6.get_xticklabels(), rotation=30, ha="right")
ax6.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax6.set_title("(f)")

fig.set_size_inches(10, 16)
fig.tight_layout(pad=1)
plt.savefig(f"{project_root}/plots/bulk_fluxes.png", dpi=300)
plt.show()
