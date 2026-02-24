import pandas as pd
import numpy as np
import keras
import keras_tuner
from keras import layers
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from src.utils import get_project_root
from src.bulk_variables.bulk_models import get_train_val_test, air_temperature_linear

project_root = get_project_root()
data_path = f"{project_root}/data"

df = pd.read_csv(f"{data_path}/training_dataset.csv")
spot_ids_vented = ["31081C", "31084C", "31085C"]
df = df.loc[df["spot_id"].isin(spot_ids_vented)]
df = air_temperature_linear(df)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.reset_index(drop=True)

features = [
    "estimated_air_temperature",
    "air_temperature",
    "solar_voltage",
    "sea_surface_temperature",
    "U_10m_mean",
]


target = "air_temperature_surface"
all_cols = features + [target]
df = df.dropna(subset=all_cols)
df = df.reset_index(drop=True)

train_idx, val_idx, test_idx = get_train_val_test(df)

# Train on residual
X_train = df.loc[train_idx, features]
y_train = df.loc[train_idx, "estimated_air_temperature"] - df.loc[train_idx, target]
X_val = df.loc[val_idx, features]
y_val = df.loc[val_idx, "estimated_air_temperature"] - df.loc[val_idx, target]
X_test = df.loc[test_idx, features]
y_test = df.loc[test_idx, "estimated_air_temperature"] - df.loc[test_idx, target]


def normalize(data):
    return (data - np.nanmean(X_train, axis=0)) / np.nanstd(X_train, axis=0)


X_train_norm = normalize(X_train)
X_val_norm = normalize(X_val)


# %% Parameter sweep
def define_model(units, num_layers, activation, lr, dropout_rate):
    model_layers = []
    for _ in range(num_layers):
        model_layers.append(
            layers.Dense(
                units,
                activation=activation,
                kernel_initializer=keras.initializers.HeNormal(),
            )
        )
        model_layers.append(layers.Dropout(dropout_rate))
    model_layers.append(layers.Dense(1))
    model = keras.Sequential(model_layers)
    model.compile(loss="mse", optimizer=Adam(learning_rate=lr))

    return model


def build_model(hp):
    units = hp.Choice("units", [4, 5, 6, 8, 16, 32, 64, 128])
    activation = hp.Choice("activation", ["relu", "swish"])
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    num_layers = hp.Choice("num_layers", [2, 3, 4, 6, 8])

    # call existing model-building code with the hyperparameter values.
    model = define_model(units=units, num_layers=num_layers, activation=activation, lr=lr, dropout_rate=dropout_rate)
    return model


tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=128,
    executions_per_trial=3,
    overwrite=False,
    directory="models/model_sweep_1",
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
    TensorBoard("models/model_sweep_1/tb_logs"),
]

tuner.search(
    X_train_norm,
    y_train,
    epochs=500,
    batch_size=256,
    verbose=0,
    validation_data=(X_val_norm, y_val),
    callbacks=callbacks,
)

tuner.results_summary()
