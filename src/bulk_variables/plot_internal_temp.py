"""
Generates Figure 3a
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.utils import get_project_root

params = {
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "axes.grid": False,
}
plt.rcParams.update(params)


project_root = get_project_root()
data_path = f"{project_root}/data"
df = pd.read_csv(f"{data_path}/bulk_variable_dataset.csv")
spot_ids_vented = ["31081C", "31084C", "31085C"]
df = df.loc[df["spot_id"].isin(spot_ids_vented)]
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.groupby("time", as_index=False)[
    ["air_temperature_surface", "air_temperature", "estimated_air_temperature"]
].mean()
mask = (df["time"] > "2023-09-14") & (df["time"] < "2023-09-24")

# %%
fig, ax = plt.subplots()
ax.plot(
    df["time"].values[mask],
    df["air_temperature_surface"].values[mask],
    color="#6929c4",
    alpha=0.8,
    linewidth=2.5,
    label="ASIT Near-Surface",
)
ax.plot(
    df["time"].values[mask],
    df["air_temperature"].values[mask],
    "-",
    color="#009d9a",
    alpha=0.8,
    linewidth=2,
    label="Spotter Internal",
)
ax.plot(
    df["time"].values[mask],
    df["estimated_air_temperature"].values[mask],
    "--",
    color="#9f1853",
    alpha=0.8,
    linewidth=1.5,
    label="Linear Regression",
)
ax.set_ylabel(r"Air Temperature ($^{\circ}$C)")
ax.legend(loc="upper left")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig.set_size_inches(9, 5)
fig.tight_layout(pad=0.5)
# plt.savefig(f"{project_root}/plots/spot_internal_temp.png", dpi=300)
plt.show()
