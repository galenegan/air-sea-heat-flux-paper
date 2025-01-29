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

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(dflux / np.nanstd(dflux), bins="auto")
ax1.vlines(x=-1.96, ymin=0, ymax=750, color="r", alpha=0.5)
ax1.vlines(x=1.96, ymin=0, ymax=750, color="r", alpha=0.5)
ax1.set_xlabel(r"$\frac{\textrm{d} Q}{\textrm{d} t}\left(\textrm{std}(\frac{\textrm{d} Q}{\textrm{d} t})\right)^{-1}$")
ax1.set_ylabel(r"Counts")

ax2.hist(dflux / np.nanstd(dflux), bins="auto")
ax2.vlines(x=-1.96, ymin=0, ymax=750, color="r", alpha=0.5)
ax2.vlines(x=1.96, ymin=0, ymax=750, color="r", alpha=0.5)
ax2.set_xlabel(r"$\frac{\textrm{d} Q}{\textrm{d} t}\left(\textrm{std}(\frac{\textrm{d} Q}{\textrm{d} t})\right)^{-1}$")
ax2.set_ylabel(r"Counts")
ax2.set_xlim(-5, 5)

fig.set_size_inches(12, 5)
fig.tight_layout(pad=0.5)
plt.savefig("plots/flux_gradient_histogram.png", dpi=300)
plt.show()