import numpy as np
import pandas as pd
from src.python.coare3p5.coare35vn import coare35vn
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter

data_path = "/Users/ea-gegan/Documents/gitrepos/asit-proxy-sensing23/src/python/scripts/analysis/publication_data_windfix_avg_dudt"
spot_ids_vented = ["31081C", "31084C", "31085C"]
df = pd.read_csv(f"{data_path}/spotter_bulk_flux_input_dataset_vented.csv")
df = df.loc[df["spot_id"].isin(spot_ids_vented)]
df["latent_heat_flux_spotter_coare"] = np.nan
df["sensible_heat_flux_spotter_coare"] = np.nan
df["ustar_spotter_coare"] = np.nan
df["U_10m_mean"] = df[["U_10m_s22", "U_10m_im"]].mean(axis=1)

# # # Filter
# fs = 1 / (20 * 60)
# fc = 1 / (720 * 60)
# b, a = butter(2, fc / (fs / 2))
# # Filtering per unit
# df["tfilt"] = np.nan
# df["qfilt"] = np.nan
# for spot_id in spot_ids_vented:
#     dfs = df.loc[df["spot_id"] == spot_id]
#     dfs["estimated_air_temperature_nn"] = filtfilt(b, a, dfs["estimated_air_temperature_nn"])
#     dfs["estimated_specific_humidity_nn"] = filtfilt(b, a, dfs["estimated_specific_humidity_nn"])
#     df.loc[dfs.index] = dfs

filenames = [
    'final_dataset.csv',
    'final_dataset_asit_u.csv',
    'final_dataset_asit_t.csv',
    'final_dataset_asit_q.csv',
    'final_dataset_asit_q_asit_t.csv',
    'final_dataset_asit_u_asit_t.csv',
    'final_dataset_asit_u_asit_q.csv'
]

for filename in filenames:
    if "asit_u" in filename:
        wind_var = "U_10m_coare"
    else:
        wind_var = "U_10m_mean"

    if "asit_t" in filename:
        temp_var = "air_temperature_surface"
    else:
        temp_var = "estimated_air_temperature_nn"

    if "asit_q" in filename:
        q_var = "specific_humidity_surface"
    else:
        q_var = "estimated_specific_humidity_nn"


    required_inputs = [
        wind_var,
        temp_var,
        q_var,
        "sea_surface_temperature",
        "atmospheric_pressure",
        "inferred_solar_radiation",
        "cp_coare_input",
        "significant_wave_height"
    ]

    for ii in df.index:

        if any(pd.isna(df.loc[ii, required_inputs])):
            continue
        out = coare35vn(
            u=df.loc[ii, wind_var],
            t=df.loc[ii, temp_var],
            q=df.loc[ii, q_var],
            ts=df.loc[ii, "sea_surface_temperature"],
            P=df.loc[ii, "atmospheric_pressure"],
            Rs=df.loc[ii, "inferred_solar_radiation"],
            zu=10,
            zt=0.2,
            zq=0.2,
            lat=41.325017,
            rain=0,
            cp=df.loc[ii, "cp_coare_input"],
            sigH=df.loc[ii, "significant_wave_height"],
        )

        df.loc[ii, "ustar_spotter_coare"] = out[0, 0]
        df.loc[ii, "sensible_heat_flux_spotter_coare"] = out[0, 2]
        df.loc[ii, "latent_heat_flux_spotter_coare"] = out[0, 3]

    df.to_csv(f"{data_path}/error_stats/{filename}", index=False)