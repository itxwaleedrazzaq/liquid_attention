import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras import layers, models, optimizers, losses, metrics
from itertools import product

# Import your LAN layer (adjust this import if needed)
from liquid_attention import LAN

np.random.seed(42)
# -----------------------------
# Synthetic dataset
# -----------------------------
def make_synthetic_data(num_samples=2000, seq_len=20, d_model=32, num_classes=5):
    """Simple classification dataset for testing."""
    X = np.random.randn(num_samples, seq_len, d_model).astype(np.float32)
    y = (np.sum(X, axis=(1, 2)) > 0).astype(int) % num_classes
    y = tf.keras.utils.to_categorical(y, num_classes)
    return X, y

# -----------------------------
# Model builder
# -----------------------------
def build_model(d_model, num_heads, mode="exact", tau_fixed=None):
    """Build LAN model with configurable mode, heads, and tau."""
    inputs = layers.Input(shape=(None, d_model))

    lan_layer = LAN(
        d_model=d_model,
        num_heads=num_heads,
        mode=mode,
        return_sequences=False,
        activation="sigmoid"
    )

    # Optionally fix tau to a constant
    if tau_fixed is not None:
        original_compute_phi_tau = lan_layer.compute_phi_tau  # store original method

        def fixed_compute_phi_tau(q, k, t):
            phi, tau, time = original_compute_phi_tau(q, k, t)
            tau = tf.ones_like(tau) * tau_fixed
            return phi, tau, time

        lan_layer.compute_phi_tau = fixed_compute_phi_tau

    x = lan_layer(inputs)
    outputs = layers.Dense(5, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()]
    )
    return model

# -----------------------------
# Experiment runner
# -----------------------------
def run_ablation():
    X, y = make_synthetic_data()

    d_model = 32
    head_counts = [1, 2, 4, 8]
    tau_values = [0.001, 0.1, 1.0, 10.0]  # None = learned
    modes = ["steady", "euler", "exact"]

    results = []

    for mode, num_heads, tau_fixed in product(modes, head_counts, tau_values):
        print(f"\n=== Running: mode={mode}, heads={num_heads}, tau_fixed={tau_fixed} ===")
        model = build_model(d_model, num_heads, mode, tau_fixed)

        history = model.fit(
            X, y,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            verbose=1
        )

        acc = history.history["val_categorical_accuracy"][-1]
        loss = history.history["val_loss"][-1]
        print(f"Validation accuracy: {acc:.4f}, loss: {loss:.4f}")

        results.append({
            "mode": mode,
            "num_heads": num_heads,
            "tau_fixed": "learned" if tau_fixed is None else tau_fixed,
            "val_acc": float(acc),
            "val_loss": float(loss)
        })

    return results

# -----------------------------
# Plotting utilities
# -----------------------------
def plot_results(results):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('seaborn-v0_8-deep')

    modes = sorted(set(r["mode"] for r in results))
    head_counts = sorted(set(r["num_heads"] for r in results))
    tau_values = sorted(set(r["tau_fixed"] for r in results), key=lambda x: (x == "learned", x))

    # -------------------------------
    # Plot 1: Accuracy vs Heads
    # -------------------------------
    plt.figure(figsize=(8, 6))
    for mode in modes:
        accs = []
        for h in head_counts:
            vals = [r["val_acc"] for r in results if r["mode"] == mode and r["num_heads"] == h]
            accs.append(np.mean(vals) if vals else np.nan)
        plt.plot(head_counts, accs, marker="o", linewidth=2, label=f"{mode} mode")

    plt.title(r"(a) Accuracy vs. Number of Heads",fontweight='bold')
    plt.xlabel("Number of Heads",fontweight='bold')
    plt.ylabel("Accuracy",fontweight='bold')
    plt.legend(title="Mode")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("ablation_study1.png", dpi=1200)
    plt.show()

    # -------------------------------
    # Plot 2: Accuracy vs Tau
    # -------------------------------
    plt.figure(figsize=(8, 6))
    for mode in modes:
        accs = []
        for tau in tau_values:
            vals = [r["val_acc"] for r in results if r["mode"] == mode and r["tau_fixed"] == tau]
            accs.append(np.mean(vals) if vals else np.nan)
        plt.plot(
            [str(t) for t in tau_values],
            accs,
            marker="s",
            linewidth=2,
            label=f"{mode} mode"
        )

    plt.title(r"(b) Accuracy vs. $\omega_\tau$",fontweight='bold')
    plt.xlabel(r"$\omega_\tau$",fontweight='bold')
    plt.ylabel("Accuracy",fontweight='bold')
    plt.legend(title="Mode")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("ablation_study2.png", dpi=1200)
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    results = run_ablation()
    print("\n=== Summary ===")
    for r in results:
        print(f"mode={r['mode']:<6} heads={r['num_heads']:<2} tau={r['tau_fixed']:<7} "
              f"acc={r['val_acc']:.4f} loss={r['val_loss']:.4f}")
    plot_results(results)
