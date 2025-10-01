'''
Modified from: https://github.com/itxwaleedrazzaq/ncpsAutonomy
'''
import pickle
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras._tf_keras.keras.layers import (
    Input,
    Dense,
    Flatten,
    Dropout,
    TimeDistributed,
    Conv2D
)
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from liquid_attention import LAN


# Configuration
MODEL_NAME = "CarRacing_LAN"
FEATURE_DIR = "tf_features"
WEIGHTS_DIR = "model_weights"
STAT_DIR = "statistics"


def load_data(pickle_path="data/data.pickle"):
    """Load dataset from pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    X, y = zip(*data)
    X = np.expand_dims(np.array(X), axis=1)
    y = np.array(y).astype(dtype="uint8")

    print(f"Dataset loaded: {len(X)} samples, {len(y)} labels")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


def build_model(num_classes, input_shape=(None, 96, 96, 3)):
    """Define LAN-based model architecture."""
    inp = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(10, (3, 3), strides=2, activation="relu"))(inp)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(20, (5, 5), strides=2, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(30, (5, 5), strides=2, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Flatten())(x)
    x = LAN(d_model=64, num_heads=16, mode="exact", return_sequences=False)(x)

    x = Dense(64, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inp, outputs=out)


def main():
    # Load dataset
    X, y = load_data()
    num_classes = len(np.unique(y))

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .batch(32)
        .shuffle(10000)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    # Build and compile model
    model = build_model(num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.99,
        patience=2,
        min_lr=1e-10,
        mode="max"
    )
    checkpoint = ModelCheckpoint(
        filepath=f"{WEIGHTS_DIR}/{MODEL_NAME}.weights.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True
    )

    # Train
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=100,
        callbacks=[reduce_lr, checkpoint]
    )

    # Optional evaluation
    # model.load_weights(f"{WEIGHTS_DIR}/{MODEL_NAME}.weights.h5")
    # loss, acc = model.evaluate(test_ds)
    # print(f"Test Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
