import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


params = {
            'axes.labelsize': 18,
            'font.size': 18,
            'legend.fontsize': 14,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'text.usetex': True,
            "font.family": "serif",
            "font.serif": "Computer Modern",
            'axes.grid': False
        }
plt.rcParams.update(params)


spot_ids_vented = ["31081C", "31084C", "31085C"]
data_path = "/Users/ea-gegan/Documents/gitrepos/asit-proxy-sensing23/src/python/scripts/analysis/publication_data_windfix_avg_dudt"
df_full = pd.read_csv(f"{data_path}/shortwave_training_dataset_all.csv")
df_vented = df_full.loc[df_full["spot_id"].isin(spot_ids_vented)]
df = df_vented.loc[df_vented["time"] < "2023-12-01"]
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.groupby("time", as_index=False)[["air_temperature_surface", "air_temperature"]].mean()
mask = ((df["time"] > "2023-09-14") & (df["time"] < "2023-09-24"))

#%%
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
ax.set_ylabel(r"Air Temperature ($^{\circ}$C)")
ax.legend(loc="upper left")
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig.set_size_inches(9, 5)
fig.tight_layout(pad=0.5)
plt.savefig("plots/spot_internal.png", dpi=300)
plt.show()