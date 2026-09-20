r"""EEGNet adapted to regression (single tanh output) for MI decoding.

Exact architecture (EEGNet-8,2 topology with a 1-unit regression head):

  Input (8, 250, 1)
   1. Conv2D        F1=8,  kernel (1,64), padding=same, no bias      # temporal
   2. BatchNorm
   3. DepthwiseConv2D kernel (8,1), depth_multiplier D=2, no bias,   # spatial
                      max_norm(1.0)
   4. BatchNorm -> ELU
   5. AveragePooling2D (1,4)
   6. Dropout 0.5
   7. SeparableConv2D F2=16, kernel (1,16), padding=same, no bias
   8. BatchNorm -> ELU
   9. AveragePooling2D (1,8)
  10. Dropout 0.5
  11. Flatten
  12. Dense(1, tanh)                                                 # output in [-1,1]

Loss is MSE (the competition metric); the tanh head bounds the output to [-1,1].
"""
from __future__ import annotations

N_CHANNELS, N_SAMPLES = 8, 250
F1, D, F2 = 8, 2, 16
KERN_TEMPORAL, KERN_SEPARABLE = 64, 16
POOL1, POOL2 = 4, 8
DROPOUT = 0.5


def build_model(n_channels: int = N_CHANNELS, n_samples: int = N_SAMPLES):
    """Return the uncompiled EEGNet-regression Keras model."""
    from tensorflow import keras
    from tensorflow.keras import layers as L

    inp = keras.Input((n_channels, n_samples, 1), name="eeg")
    x = L.Conv2D(F1, (1, KERN_TEMPORAL), padding="same", use_bias=False)(inp)   # temporal
    x = L.BatchNormalization()(x)
    x = L.DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=D,   # spatial
                          depthwise_constraint=keras.constraints.MaxNorm(1.0))(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("elu")(x)
    x = L.AveragePooling2D((1, POOL1))(x)
    x = L.Dropout(DROPOUT)(x)
    x = L.SeparableConv2D(F2, (1, KERN_SEPARABLE), padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("elu")(x)
    x = L.AveragePooling2D((1, POOL2))(x)
    x = L.Dropout(DROPOUT)(x)
    x = L.Flatten()(x)
    out = L.Dense(1, activation="tanh", name="regression")(x)                    # [-1,1]
    return keras.Model(inp, out, name="EEGNet_regression")


def compile_model(model, lr: float = 1e-3):
    from tensorflow import keras
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model
