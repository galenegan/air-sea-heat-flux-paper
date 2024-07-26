import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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


def set_lims(ax, scale=1.0):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lower_lim = scale * min(xmin, ymin)
    upper_lim = scale * max(xmax, ymax)
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)


project_root = get_project_root()
data_path = f"{project_root}/data"
df = pd.read_csv(f"{data_path}/final_flux_dataset.csv")
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.loc[df["spot_id"] == "31085C"]  # Last remaining vented Spotter during most of the test set

# %%
dflux = np.gradient(df["sensible_heat_flux_dc"].values, df["epoch"].values)
gradient_cutoff = 1.96 * np.nanstd(dflux)
asit_residual = np.abs(df["sensible_heat_flux_asit_coare"].values - df["sensible_heat_flux_dc"].values)
spot_asit_residual = np.abs(df["sensible_heat_flux_asit_coare"].values - df["sensible_heat_flux_spotter_coare"].values)
asit_residual_cutoff = 75
residual_gradient = np.gradient(asit_residual, df["epoch"].values)
asit_relative_residual = np.abs(df["sensible_heat_flux_dc"].values / df["sensible_heat_flux_asit_coare"].values)

mask = (
    (df["sensible_heat_flux_dc"].values > -300)
    & (df["sensible_heat_flux_dc"].values < 300)
    & (dflux < gradient_cutoff)
    & (asit_residual < asit_residual_cutoff)
    & (asit_relative_residual < 5)
    & (df["rain_rate"].values < 1)
    & (df["epoch"].values < pd.Timestamp("2024-01-01T00:00:00Z").timestamp())
)


# %% Setting up the plot
fig = plt.figure(constrained_layout=True)
outer_grid = GridSpec(4, 2, figure=fig)
second_cell = outer_grid[1, :]
fourth_cell = outer_grid[3, :]
inner_grid_1 = GridSpecFromSubplotSpec(1, 2, second_cell, width_ratios=[0.33, 0.67], wspace=0.05)
inner_grid_2 = GridSpecFromSubplotSpec(1, 2, fourth_cell, width_ratios=[0.33, 0.67], wspace=0.05)

ax1 = fig.add_subplot(outer_grid[0, 0])
ax2 = fig.add_subplot(outer_grid[0, 1])
ax31 = fig.add_subplot(inner_grid_1[0, 0])
ax32 = fig.add_subplot(inner_grid_1[0, 1], sharey=ax31)
ax4 = fig.add_subplot(outer_grid[2, 0])
ax5 = fig.add_subplot(outer_grid[2, 1])
ax61 = fig.add_subplot(inner_grid_2[0, 0])
ax62 = fig.add_subplot(inner_grid_2[0, 1], sharey=ax61)


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
ax1.legend(loc="lower right", fontsize=12)
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
ax2.legend(loc="lower right", fontsize=12)
ax2.set_title("(b)")
set_lims(ax2)

ax31.plot(
    df["time"].values[mask],
    df["sensible_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    markersize=4,
    label="Direct Covariance",
)
ax32.plot(
    df["time"].values[mask],
    df["sensible_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    markersize=4,
    label="Direct Covariance",
)

ax31.plot(
    df["time"].values[mask],
    df["sensible_heat_flux_asit_coare"].values[mask],
    color="#6929c4",
    alpha=0.8,
    linewidth=2.5,
    label="ASIT Bulk",
)
ax32.plot(
    df["time"].values[mask],
    df["sensible_heat_flux_asit_coare"].values[mask],
    color="#6929c4",
    alpha=0.8,
    linewidth=2.5,
    label="ASIT Bulk",
)

ax31.plot(
    df["time"].values[mask],
    df["sensible_heat_flux_spotter_coare"].values[mask],
    "-",
    color="#009d9a",
    alpha=0.8,
    linewidth=2,
    label="Spotter Bulk",
)
ax32.plot(
    df["time"].values[mask],
    df["sensible_heat_flux_spotter_coare"].values[mask],
    "-",
    color="#009d9a",
    alpha=0.8,
    linewidth=2,
    label="Spotter Bulk",
)


# zoom-in / limit the view to different portions of the data
ax31.set_xlim(pd.to_datetime("2023-09-01T00:00:00Z"), pd.to_datetime("2023-10-05T00:00:00Z"))  # outliers only
ax32.set_xlim(pd.to_datetime("2023-11-01T00:00:00Z"), pd.to_datetime("2024-01-07T00:00:00Z"))  # most of the data
# hide the spines between ax and ax2
ax31.spines.right.set_visible(False)
ax32.spines.left.set_visible(False)
ax31.yaxis.tick_left()
ax31.tick_params(labelright=False)  # don't put tick labels at the right
ax32.yaxis.tick_right()

d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax31.transAxes, color="k", clip_on=False)
ax31.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax31.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax32.transAxes)  # switch to the right axes
ax32.plot((-d, +d / 10), (1 - d, 1 + d), **kwargs)
ax32.plot((-d, +d / 10), (-d, +d), **kwargs)


ax31.set_ylabel(r"Sensible Heat Flux (W/m$^2$)")
ax31.legend()
plt.setp(ax31.get_xticklabels(), rotation=30, ha="right")
plt.setp(ax32.get_xticklabels(), rotation=30, ha="right")
ax31.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax32.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax31.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
ax32.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
ax32.annotate("(c)", xy=(0.2, 1.05), xycoords="axes fraction")

# %% Now getting the latent heat flux
dflux = np.gradient(df["latent_heat_flux_dc"].values, df["epoch"].values)
gradient_cutoff = 1.96 * np.nanstd(dflux)
asit_residual = np.abs(df["latent_heat_flux_asit_coare"].values - df["latent_heat_flux_dc"].values)
asit_residual_cutoff = 100
asit_relative_residual = np.abs(df["latent_heat_flux_dc"].values / df["latent_heat_flux_asit_coare"].values)

mask = (
    (df["latent_heat_flux_dc"].values > -300)
    & (df["latent_heat_flux_dc"].values < 500)
    & (dflux < gradient_cutoff)
    & (asit_residual < asit_residual_cutoff)
    & (asit_relative_residual < 5)
    & (df["rain_rate"].values < 1)
    & (df["epoch"].values < pd.Timestamp("2024-01-01T00:00:00Z").timestamp())
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
ax4.legend(loc="lower right", fontsize=12)
ax4.set_title("(d)")
set_lims(ax4, scale=1.2)

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
ax5.legend(loc="lower right", fontsize=12)
ax5.set_title("(e)")
set_lims(ax5, scale=1.2)

ax61.plot(
    df["time"].values[mask],
    df["latent_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    markersize=4,
    label="Direct Covariance",
)
ax61.plot(
    df["time"].values[mask],
    df["latent_heat_flux_asit_coare"].values[mask],
    color="#6929c4",
    alpha=0.8,
    linewidth=2.5,
    label="ASIT Bulk",
)
ax61.plot(
    df["time"].values[mask],
    df["latent_heat_flux_spotter_coare"].values[mask],
    "-",
    color="#009d9a",
    alpha=0.8,
    linewidth=2,
    label="Spotter Bulk",
)

ax62.plot(
    df["time"].values[mask],
    df["latent_heat_flux_dc"].values[mask],
    "o",
    color="#012749",
    markersize=4,
    label="Direct Covariance",
)
ax62.plot(
    df["time"].values[mask],
    df["latent_heat_flux_asit_coare"].values[mask],
    color="#6929c4",
    alpha=0.8,
    linewidth=2.5,
    label="ASIT Bulk",
)
ax62.plot(
    df["time"].values[mask],
    df["latent_heat_flux_spotter_coare"].values[mask],
    "-",
    color="#009d9a",
    alpha=0.8,
    linewidth=2,
    label="Spotter Bulk",
)


# zoom-in / limit the view to different portions of the data
ax61.set_xlim(pd.to_datetime("2023-09-01T00:00:00Z"), pd.to_datetime("2023-10-05T00:00:00Z"))  # outliers only
ax62.set_xlim(pd.to_datetime("2023-11-01T00:00:00Z"), pd.to_datetime("2024-01-10T00:00:00Z"))  # most of the data
# hide the spines between ax and ax2
ax61.spines.right.set_visible(False)
ax62.spines.left.set_visible(False)
ax61.yaxis.tick_left()
ax61.tick_params(labelright=False)  # don't put tick labels at the right
ax62.yaxis.tick_right()
ax61.set_ylim(-175, 400)

d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax61.transAxes, color="k", clip_on=False)
ax61.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax61.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax62.transAxes)  # switch to the right axes
ax62.plot((-d, +d / 10), (1 - d, 1 + d), **kwargs)
ax62.plot((-d, +d / 10), (-d, +d), **kwargs)


ax61.set_ylabel(r"Sensible Heat Flux (W/m$^2$)")
ax62.legend()
plt.setp(ax61.get_xticklabels(), rotation=30, ha="right")
plt.setp(ax62.get_xticklabels(), rotation=30, ha="right")
ax61.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax62.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax61.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
ax62.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
ax62.annotate("(f)", xy=(0.2, 1.05), xycoords="axes fraction")


fig.set_size_inches(10, 16)
fig.tight_layout(pad=1)
plt.savefig(f"{project_root}/plots/bulk_fluxes.png", dpi=300)
plt.show()
