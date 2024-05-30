import pandas as pd
import numpy as np
import keras
import keras_tuner
from keras import layers, regularizers
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from src.utils import get_project_root
from src.bulk_variables.bulk_models import get_train_val_test

project_root = get_project_root()
data_path = f"{project_root}/data"

df = pd.read_csv(f"{data_path}/training_dataset.csv")
spot_ids_vented = ["31081C", "31084C", "31085C"]
df = df.loc[df["spot_id"].isin(spot_ids_vented)]
df = df.reset_index(drop=True)

# This is set up for air temperature, but the features will vary depending on the model
# See bulk_models.py for details
features = [
    "air_temperature",
    "solar_voltage",
    "log_battery_power",
    "sea_surface_temperature",
    "estimated_air_temperature",
    "atmospheric_pressure",
    "relative_humidity",
    "U_10m_mean",
    "significant_wave_height",
]
target = "air_temperature_surface"
all_cols = features + [target]
df = df.dropna(subset=all_cols)
df = df.reset_index(drop=True)
X = df[features].values
y = df["estimated_air_temperature"].values - df[target].values  # Train on residual

# Train/val/test split
train_idx, val_idx, test_idx = get_train_val_test(df)

X_train = X[train_idx, :]
y_train = y[train_idx]
X_eval = X[val_idx, :]
y_eval = y[val_idx]


def normalize(data):
    return (data - np.nanmean(X_train, axis=0)) / np.nanstd(X_train, axis=0)


X_train_norm = normalize(X_train)
X_eval_norm = normalize(X_eval)


# %% Parameter sweep
def define_model(units, num_layers, activation, lr, l2):
    model_layers = [
        layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizers.L2(l2=l2),
            kernel_initializer=keras.initializers.HeNormal(),
        )
    ] * num_layers
    model_layers += [layers.Dense(1)]
    model = keras.Sequential(model_layers)
    model.compile(loss="mse", optimizer=Adam(learning_rate=lr))

    return model


def build_model(hp):
    units = hp.Choice("units", [2, 4, 6, 8, 16, 32])
    activation = hp.Choice("activation", ["relu", "swish"])
    lr = hp.Float("lr", min_value=1e-5, max_value=1e-1, sampling="log")
    l2 = hp.Float("l2", min_value=1e-5, max_value=1e-1, sampling="log")
    num_layers = hp.Choice("num_layers", [2, 3, 4, 6, 8])

    # call existing model-building code with the hyperparameter values.
    model = define_model(units=units, num_layers=num_layers, activation=activation, lr=lr, l2=l2)
    return model


tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=128,
    executions_per_trial=1,
    overwrite=False,
    directory="models/model_sweep_0",
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, start_from_epoch=250),
    TensorBoard("models/model_sweep_0/tb_logs"),
]

tuner.search(X_train_norm, y_train, epochs=500, verbose=0, validation_data=(X_eval_norm, y_eval), callbacks=callbacks)

tuner.results_summary()
