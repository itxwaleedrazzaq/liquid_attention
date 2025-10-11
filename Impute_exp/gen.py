import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.layers import TimeDistributed,Input,RepeatVector,Dense
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.optimizers import RMSprop
from liquid_attention import LAN

# === 1. Generate synthetic time series ===
np.random.seed(42)
time = np.arange(0, 500, 0.01)
values = np.sin(time) + np.random.normal(0, 0.05, len(time))
df = pd.DataFrame({"time": time, "value": values})

# === 2. Randomly remove values ===
missing_rate = 0.05
mask = np.random.rand(len(df)) < missing_rate
df.loc[mask, "value"] = np.nan

# === 3. Scale data ===
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["value"]])
scaled = np.where(np.isnan(scaled), 0, scaled)  # replace NaN with zero for model input

# === 4. Create input sequences ===
window_size = 64
X = np.lib.stride_tricks.sliding_window_view(scaled, (window_size, 1))[:, 0, :, :]
# shape: (num_sequences, window_size, 1)

# === 5.LAN Autoencoder ===
model = Sequential([
    Input(shape=(window_size, 1)),
    LAN(d_model=32, num_heads=8, activation='relu', return_sequences=True),
    LAN(d_model=16, num_heads=8, activation='relu', return_sequences=False),
    RepeatVector(window_size),
    LAN(d_model=16, num_heads=8, activation='relu', return_sequences=True),
    LAN(d_model=32, num_heads=8, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])

model.compile(
    optimizer=RMSprop(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

model.fit(X, X, epochs=10, batch_size=32, verbose=1, validation_split=0.1)

# === 6. Predict reconstructed sequences ===
reconstructed = model.predict(X, verbose=0).squeeze()

# === 7. Merge overlapping predictions ===
reconstructed_full = np.zeros(len(df))
count = np.zeros(len(df))

for i in range(reconstructed.shape[0]):
    reconstructed_full[i:i + window_size] += reconstructed[i]
    count[i:i + window_size] += 1

reconstructed_full /= np.maximum(count, 1)  # prevent division by zero
imputed = scaler.inverse_transform(reconstructed_full.reshape(-1, 1)).flatten()

# === 8. Fill missing values ===
df["imputed"] = df["value"].fillna(pd.Series(imputed, index=df.index))

# === 9. Plot only test data ===
test_split = 0.95  # last 20% of data as test
test_start = int(len(df) * test_split)
test_df = df.iloc[test_start:]
test_mask = mask[test_start:]
test_imputed = imputed[test_start:]

plt.figure(figsize=(12, 6))
plt.style.use('seaborn-v0_8-deep')

# Plot ground truth (solid line)
plt.plot(test_df["time"], test_df["value"], label="Ground Truth", color="blue", alpha=0.7)

# Plot missing values (dots)
plt.scatter(
    test_df["time"][test_mask],
    test_imputed[test_mask],
    color="red",
    s=20,
    label="Missing Values"
)

# Plot imputed output (dashed line)
plt.plot(
    test_df["time"],
    test_df["imputed"],
    linestyle="--",
    color="green",
    alpha=0.9,
    label="Imputed Sequence"
)

plt.title("Seq2Seq Time Series Imputation")
plt.xlabel("Time")
plt.grid(True, linestyle="--", alpha=0.5)
plt.ylabel("Value")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
