import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from src.coare3p5.coare35vn import coare35vn
from src.utils import get_project_root

project_root = get_project_root()
data_path = f"{project_root}/data"
out_path = "./flux_data"
df = pd.read_csv(f"{data_path}/bulk_variable_dataset.csv")
df["latent_heat_flux_spotter_coare"] = np.nan
df["sensible_heat_flux_spotter_coare"] = np.nan
df["ustar_spotter_coare"] = np.nan

df["estimated_air_temperature_filtered"] = median_filter(df["estimated_air_temperature_nn"], size=15)
df["estimated_specific_humidity_filtered"] = median_filter(df["estimated_specific_humidity_nn"], size=15)

filenames = [
    "final_flux_dataset.csv",
    "final_flux_dataset_filtered_inputs.csv",
]

for filename in filenames:
    wind_var = "U_10m_mean"
    if "filtered_inputs" in filename:
        temp_var = "estimated_air_temperature_filtered"
        q_var = "estimated_specific_humidity_filtered"
    else:
        temp_var = "estimated_air_temperature_nn"
        q_var = "estimated_specific_humidity_nn"

    required_inputs = [
        wind_var,
        temp_var,
        q_var,
        "sea_surface_temperature",
        "atmospheric_pressure",
        "estimated_shortwave_rf",
        "cp_coare_input",
        "significant_wave_height",
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
            Rs=df.loc[ii, "estimated_shortwave_rf"],
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

    df.to_csv(f"{out_path}/{filename}", index=False)
