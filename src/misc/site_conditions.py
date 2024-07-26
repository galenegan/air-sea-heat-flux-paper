import matplotlib.pyplot as plt
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


project_root = get_project_root()
data_path = f"{project_root}/data"
df = pd.read_csv(f"{data_path}/training_dataset.csv")
df["time"] = pd.to_datetime(df["time"], utc=True)
cols = [
    "time",
    "air_temperature_surface",
    "specific_humidity_surface",
    "sea_surface_temperature",
    "U_10m_coare",
    "significant_wave_height",
    "mean_period",
    "mean_direction",
]

df = df.groupby("epoch", as_index=False)[cols].mean()
# %% Plotting
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True)
ax1.plot(df["time"], df["air_temperature_surface"], "o", markersize=2, color="#012749")
ax1.set_ylabel(r"$T_{\textrm{air}}$ ($^{\circ}$C)")
ax1.set_yticks(np.arange(-10, 31, 10))

ax2.plot(df["time"], df["specific_humidity_surface"], "o", markersize=2, color="#012749")
ax2.set_ylabel(r"$q_{\textrm{air}}$ (g/kg)")
ax2.set_yticks(np.arange(0, 22, 5))

ax3.plot(df["time"], df["U_10m_coare"], "o", markersize=2, color="#012749")
ax3.set_ylabel(r"$U_{10}$ (m/s)")
ax3.set_yticks(np.arange(0, 26, 5))

ax4.plot(df["time"], df["sea_surface_temperature"], "o", markersize=2, color="#012749")
ax4.set_ylabel(r"$T_{\textrm{water}}$ ($^{\circ}$C)")
ax4.set_yticks(np.arange(0, 26, 5))

ax5.plot(df["time"], df["significant_wave_height"], "o", markersize=2, color="#012749")
ax5.set_ylabel(r"$H_{\textrm{sig}}$ (m)")
ax5.set_yticks(np.arange(0, 8, 2.5))

ax6.plot(df["time"], df["mean_period"], "o", markersize=2, color="#012749")
ax6.set_ylabel(r"$T_m$ (s)")
ax6.set_yticks(np.arange(0, 13, 4))

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.grid()

fig.set_size_inches(18, 18)
fig.autofmt_xdate()
plt.rcParams.update(params)
plt.savefig(f"{project_root}/plots/site_conditions.png", dpi=300, bbox_inches="tight")
plt.show()
