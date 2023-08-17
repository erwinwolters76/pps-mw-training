import tensorflow as tf  # type: ignore
from tensorflow.keras import layers  # type: ignore


class SymmetricPadding(layers.Layer):
    """Symmetric padding."""

    def __init__(self, amount: int):
        super().__init__()
        self.paddings = tf.constant(
            [[0, 0], [amount, amount], [amount, amount], [0, 0]]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.pad(x, self.paddings, "SYMMETRIC")


class UpSampling2D(layers.Layer):
    """Upsampling layer by bilinear interpolation."""

    def __init__(self):
        super().__init__()
        self.upsample = layers.UpSampling2D(
            size=(2, 2),
            interpolation="bilinear",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.upsample(x)
