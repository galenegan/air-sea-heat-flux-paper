# Air-sea Heat Flux Observations from a Spotter Buoy
This repo contains all code necessary to reproduce the results presented in "Air-sea Heat Flux Observations from a Spotter Buoy," which has been submitted for publication by Galen Egan, Seth Zippel, and Pieter Smit. 


## Interpreter Setup
The code herein was executed using a Python 3.9.10 interpreter with package versions specified in `requirements.txt`

## Source Data
The source dataset can be found at [this link](https://oregonstate.box.com/s/7sw3lkku63s3jxqrenzx7jjmnc1snprm). After
downloading the data, it should be placed in the `data` folder in the project home directory.

## Data Processing and Figure Generation Scripts

The semi-analytical air temperature calibration in `bulk_variables/calibrate_analytical_air_temp.py` and the plotting scripts in `src/supplemental` require only the source training dataset. However, other scripts must be run in a particular order to generate data files required for other scripts. In particular:

1. `src/bulk_variables/plot_and_export_bulk_variables.py` will create a plot of all bulk variable estimates and save a new csv at `data/bulk_variable_dataset.csv`
2. After the bulk variable dataset is generated, `src/bulk_fluxes/calculate_bulk_fluxes.py` will save a few variations of `data/final_flux_dataset.csv`, some with Spotter-estimated variables replaced by their ASIT equivalents
3. After the final flux datasets are generated, `src/bulk_fluxes/generate_error_stats_table.py` and `src/bulk_fluxes/plot_bulk_fluxes.py` can be run to generate Table 1 and Figure 3, respectively. 
