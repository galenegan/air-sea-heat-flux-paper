import numpy as np
import pandas as pd
from src.coare3p5.coare35vn import coare35vn
from src.utils import get_project_root

project_root = get_project_root()
data_path = f"{project_root}/data"
df = pd.read_csv(f"{data_path}/bulk_variable_dataset.csv")
df["latent_heat_flux_spotter_coare"] = np.nan
df["sensible_heat_flux_spotter_coare"] = np.nan
df["ustar_spotter_coare"] = np.nan

filenames = [
    "final_flux_dataset.csv",
    "final_flux_dataset_asit_u.csv",
    "final_flux_dataset_asit_t.csv",
    "final_flux_dataset_asit_q.csv",
    "final_flux_dataset_asit_q_asit_t.csv",
    "final_flux_dataset_asit_u_asit_t.csv",
    "final_flux_dataset_asit_u_asit_q.csv",
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

    df.to_csv(f"{data_path}/{filename}", index=False)
