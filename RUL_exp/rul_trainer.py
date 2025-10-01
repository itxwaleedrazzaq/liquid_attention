import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras._tf_keras.keras.layers import Input, Dense, Reshape, Conv1D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import AdamW
from ..liquid_attention.tf import LAN
from utils.preprocess import process_features

# Configuration
MODEL_NAME = "RUL_LAN_Steady"
FEATURE_DIR = "tf_features_pronostia"
WEIGHTS_DIR = "model_weights"
STATS_DIR = "statistics"

BEARINGS = ["Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"]

# -----------------------------
# Load and preprocess data
# -----------------------------s
dfs = [pd.read_csv(f"{FEATURE_DIR}/{bearing}_features.csv") for bearing in BEARINGS]

# Vibration features
horizontal_data = [np.array(df["Horizontal"].apply(eval).tolist()) for df in dfs]
X_h = np.vstack([process_features(data) for data in horizontal_data])

vertical_data = [np.array(df["Vertical"].apply(eval).tolist()) for df in dfs]
X_v = np.vstack([process_features(data) for data in vertical_data])

vibration_features = np.concatenate((X_h, X_v), axis=-1)

# Other features
t_data = np.concatenate([df["Time"].values.reshape(-1, 1) for df in dfs], axis=0)
T_data = np.concatenate([(df["Temperature"].values + 273.15).reshape(-1, 1) for df in dfs], axis=0)
y = np.concatenate([df["Degradation"].values.reshape(-1, 1) for df in dfs], axis=0)
RPM = np.concatenate([df["RPM"].values.reshape(-1, 1) for df in dfs], axis=0)
Load = np.concatenate([df["Load"].values.reshape(-1, 1) for df in dfs], axis=0)

# Combine features
X = np.concatenate([vibration_features, t_data, T_data], axis=1)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Model
# -----------------------------
def build_model(input_shape=(16,)):
    """Build LAN model for Remaining Useful Life prediction."""
    inputs = Input(shape=input_shape)
    x = Reshape((1, input_shape[0]))(inputs)
    x = Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = Conv1D(16, kernel_size=2, padding="same", activation="relu")(x)
    x = LAN(
        d_model=16,
        num_heads=8,
        mode="euler",
        delta_t=0.5,
        euler_steps=20,
        return_sequences=False,
    )(x)
    outputs = Dense(1, activation="linear")(x)
    return Model(inputs, outputs)


model = build_model()
model.summary()

model.compile(
    optimizer=AdamW(learning_rate=0.001),
    loss="mse",
    metrics=["mae"],
)

# -----------------------------
# Training
# -----------------------------
history = model.fit(
    X, y,
    epochs=150,
    validation_split=0.2,
)

# Save trained model (optional)
model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")
