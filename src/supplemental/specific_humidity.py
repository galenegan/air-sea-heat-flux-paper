import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.coare3p5.meteo import qair
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
mask = df["time"] < "2023-12-01"

qair_inner, _ = qair(df["air_temperature"], df["atmospheric_pressure"], df["relative_humidity"])
qair_outer = df["specific_humidity_surface"]
solar_down = df["solar_down_24m"]

# %%
fig, ax = plt.subplots()
im = ax.scatter(qair_inner, qair_outer, c=solar_down, marker="o")
cb = fig.colorbar(im, ax=ax, extend="max")
cb.ax.set_title(r"$Q_{\textrm{sw,down}}$ (W/m$^2$)")
ax.set_xlabel(r"$q$ internal (g/kg)")
ax.set_ylabel("$q$ atmospheric (g/kg)")
fig.set_size_inches(7, 5)
fig.tight_layout(pad=1)
plt.savefig(f"{project_root}/plots/q_supplemental.png", dpi=300)
plt.show()
