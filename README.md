# Liquid Attention Network (LAN)

This repository contains implementations and experiments for the **Liquid Attention Network (LAN)** developed at Con [update with full name or link later].

---

## Experiments

### 1. Universal Approximation Verification

Demonstration and verification of LAN’s universal approximation capability.

![Universal Approximation Demo](assets/universal_approximation.gif)
*Replace `assets/universal_approximation.gif` with your GIF file path.*

---

### 2. Event-based MNIST

Training and evaluation on the event-based MNIST dataset.

* Code available in: `mnist_exp/`

```bash
# Example: how to run training
python train_mnist.py
```

---

### 3. Person Activity Recognition (PAR)

Activity recognition experiment implementation.

* Code available in: `PAR_exp/`

```bash
# Example: how to run
python run_par.py
```

---

### 4. Lane Keeping for Autonomous Vehicles

**a) CarRacing (`carracing_exp/`)**

```bash
python drive.py
```

**b) Udacity Simulator (`Udacity_exp/`)**

1. Run the simulator in Autonomous mode.
2. Execute:

```bash
python drive.py model_weights/Udacity_LAN.keras
```

---

### 5. Remaining Useful Life (RUL) Estimation

Dataset and code for RUL estimation experiments.

* Code and data available in: `RUL_exp/`

```bash
# Example: how to run
python run_rul.py
```

---

## Folder Structure

```
├── mnist_exp/         # Event-based MNIST experiment code
├── PAR_exp/           # Person Activity Recognition experiment code
├── carracing_exp/     # Lane keeping CarRacing experiment code
├── Udacity_exp/       # Lane keeping Udacity Simulator code
├── RUL_exp/           # Remaining Useful Life experiment code
└── assets/            # GIFs and images
```

---

## License

[Specify your license here, e.g., MIT License]
