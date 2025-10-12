# Liquid Attention Network (LAN)

Attention and its variants have excelled in sequential learning by focusing on the most relevant parts of a sequence, unlike traditional RNNs that process inputs step-by-step and suffer from vanishing gradients. However, standard attention operates in discrete time, limiting its ability to model continuous-time trajectories often captured by continuous-time recurrent neural networks (CT-RNNs). Inspired by biologically motivated Liquid Neural Networks (LNNs), we introduce the _Liquid Attention Network_ (LAN), where attention logits are modeled as solutions to a linear ODE modulated by nonlinear interlinked gates. This formulation allows logits to evolve dynamically with input-dependent (liquid) time constants. We propose three modes for training LAN using backpropagation through time (BPTT): (i) _Euler_, updating logits via explicit Euler integration, (ii) _Exact_, computing the analytical ODE solution, and (iii) _Steady_, using the steady-state solution analogous to standard scaled dot-product attention. Theoretically, LAN is a logit-state stable, error-bounded, and universal function approximator. Empirically, LAN consistently matches or outperforms state-of-the-art baselines on irregularly sampled, multidisciplinary time-series tasks.
---

## LAN Model Example

```python
import tensorflow as tf
from liquid_attention import LAN

# Create the model
inputs = tf.keras.Input(shape=(1, 1))
attn = LAN(
    d_model=64,
    num_heads=16,
    mode='exact, #steady or euler
    delta_t=0.1,
    euler_steps=10,
    activation='sigmoid',
    return_sequences=False,
    return_attention=False
)(inputs)
outputs = tf.keras.layers.Dense(1)(attn)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```


## Experiments

### 1. Universal Approximation Verification

Demonstration and verification of LAN’s universal approximation capability.

![Universal Approximation Demo](plots/uat.gif)

---

### 2. Event-based MNIST

Training and evaluation on the event-based MNIST dataset.

* Code available in: `mnist_exp/`

```bash
python mnist_trainer.py
```

---

### 3. Person Activity Recognition (PAR)

Activity recognition experiment implementation.

* Code available in: `PAR_exp/`

```bash
python PAR_trainer.py
```

---

### 4. Lane Keeping for Autonomous Vehicles

**a) CarRacing (`CarRacing_exp/`)**

```bash
python drive.py
```
[Watch the demo on YouTube](https://youtu.be/PAclVXbzsms)


**b) Udacity Simulator (`Udacity_exp/`)**

1. Run the simulator in Autonomous mode.
2. Execute:

```bash
python drive.py model_weights/Udacity_LAN.keras
```
[Watch the demo on YouTube](https://youtu.be/tKfO55TwN0M)

---

### 5. Remaining Useful Life (RUL) Estimation

Dataset and code for RUL estimation experiments.

* Code and data available in: `RUL_exp/`

```bash
# Example: how to run
python rul_trainer.py
```


---

## License

[Specify your license here, e.g., MIT License]
