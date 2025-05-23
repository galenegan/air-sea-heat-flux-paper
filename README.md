# Observing bulk meteorological parameters and air-sea heat fluxes with a Spotter buoy

This repo contains all code necessary to reproduce the results presented in "Observing bulk meteorological parameters
and air-sea heat fluxes with a Spotter buoy," which has been submitted for publication by Galen Egan, Seth Zippel, and
Pieter Smit.

## Interpreter Setup

The code herein was executed using a Python 3.9 interpreter with package versions specified in `requirements.txt`

## Source Data

The source dataset can be found at [this link](https://oregonstate.box.com/s/s33nou19xzbjvcmvifkmdyov9mmtycqh). After
downloading the data, it should be placed in the `data` folder in the project home directory.

## Data Processing and Figure Generation Scripts

The `site_conditions.py` script in `misc/` requires only the source training dataset. However, other scripts must be run
in a particular order to generate data files required for other scripts. For example:

1. `src/bulk_variables/plot_and_export_bulk_variables.py` will create a plot of all bulk variable estimates and save a
   new csv at `data/bulk_variable_dataset.csv`
2. After the bulk variable dataset is generated, `src/bulk_fluxes/calculate_bulk_fluxes.py` will save a few variations
   of `data/final_flux_dataset.csv`, some with Spotter-estimated variables replaced by their ASIT equivalents
3. After the final flux datasets are generated, `src/bulk_fluxes/generate_error_stats_table.py`
   and `src/bulk_fluxes/plot_bulk_fluxes.py` can be run to generate Table 2 and Figure 7, respectively.

All of the scripts in the repo that generate either a table or figure in the paper have a comment string at the top of
the file stating which content it generates.

## ML Models

The models described in the paper are stored in `src/bulk_variables/models/{variable_name}/{model_type}`. Examples of
how to apply the models are in their corresponding utility functions in `src/bulk_variables/bulk_models.py`. The only
model that is not described in the manuscript is `air_temp/nn_full_dataset`, which is a neural network trained on the
entire training dataset that we collected. This would be the recommended model for inferring air temperature from an
arbitrary Spotter buoy from the global network. 