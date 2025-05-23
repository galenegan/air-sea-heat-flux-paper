"""
Generates Figure 5
"""

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


def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit + 1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled


def mae(x, y):
    return np.nanmean(np.abs(x - y))


def rmse(x, y):
    return np.sqrt(np.nanmean((x - y) ** 2))


def bias(x, y):
    return np.nanmean(x - y)


project_root = get_project_root()
data_path = f"{project_root}/data"
df = pd.read_csv(f"{data_path}/bulk_variable_dataset.csv")
df = df.loc[df["spot_id"] == "31085C"]
df["time"] = pd.to_datetime(df["time"])

x = np.load(f"{project_root}/src/bulk_variables/models/q/exponential_decay/optimal_params.npy", allow_pickle=True)
a, b, c = x[0], x[1], x[2]
df["days"] = (df["epoch"] - df["epoch"].min()) / 86400
exp_fit = a * np.exp(b * df["days"].values) + c

q_air = interpolate_gaps(df["specific_humidity_surface"].values, limit=10)
q_int = interpolate_gaps(df["q_inner"].values, limit=10)
mae_fit = mae(exp_fit, q_air - q_int)
rmse_fit = rmse(exp_fit, q_air - q_int)
bias_fit = bias(exp_fit, q_air - q_int)
# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(df["time"], q_air, "o", markersize=3, color="#012749", label=r"ASIT $q_{\textrm{air}}$")
ax1.plot(df["time"], q_int, "o", markersize=3, alpha=0.1, color="#009d9a", label=r"Spotter Internal $q_{\textrm{int}}$")
leg = ax1.legend()
ax1.set_ylabel("$q$ (g/kg)")
ax1.set_title("(a)")

for lh in leg.legendHandles[1:]:
    lh.set_alpha(0.5)

ax2.plot(df["time"], q_air - q_int, "o", markersize=3, color="#012749")
ax2.plot(
    df["time"],
    exp_fit,
    linewidth=3,
    color="#9f1853",
    label=f"Exp Fit, MAE = {mae_fit:.2f}, RMSE = {rmse_fit:.2f}, Bias = {bias_fit:.2f} g/kg",
)
ax2.set_ylabel(r"$q_{\textrm{air}} - q_{\textrm{int}}$ (g/kg)")
ax2.set_title("(b)")
ax2.legend()
fig.autofmt_xdate()
fig.set_size_inches(14, 5)
fig.tight_layout(pad=1)
# plt.savefig(f"{project_root}/plots/q.png", dpi=300)
plt.show()
