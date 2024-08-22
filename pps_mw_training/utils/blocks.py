import tensorflow as tf  # type: ignore
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

from pps_mw_training.utils.layers import SymmetricPadding, UpSampling2D


class ConvolutionBlock(layers.Layer):
    """
    A convolution block consisting of a pair of 3x3 convolutions
    followed by batch normalization and ReLU activations.
    """

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.block = keras.Sequential()
        self.block.add(SymmetricPadding(1))
        self.block.add(
            layers.Conv2D(
                channels_out,
                3,
                padding="valid",
                input_shape=(None, None, channels_in),
            )
        )
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())
        # self.block.add(layers.Dropout(0.3))  # Added dropout here
        self.block.add(SymmetricPadding(1))
        self.block.add(layers.Conv2D(channels_out, 3, padding="valid"))
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())
        # self.block.add(layers.Dropout(0.3))  # Added second dropout

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.block(x)


class DownsamplingBlock(keras.Sequential):
    """
    A downsampling block consisting of a max pooling layer and a
    convolution block.
    """

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.add(layers.MaxPooling2D(strides=(2, 2)))
        self.add(ConvolutionBlock(channels_in, channels_out))


class UpsamplingBlock(layers.Layer):
    """
    An upsampling block which which uses bilinear interpolation
    to increase the resolution. This is followed by a 1x1 convolution to
    reduce the number of channels, concatenation of the skip inputs
    from the corresponding downsampling layer and a convolution block.
    """

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.upsample = UpSampling2D()
        self.reduce_channels = layers.Conv2D(
            channels_in // 2, 1,
            padding="same",
            input_shape=(None, None, channels_in),
        )
        self.concat = layers.Concatenate()
        self.conv_block = ConvolutionBlock(channels_in, channels_out)

    def call(self, xs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x, x_skip = xs
        x = self.upsample(x)
        x = self.reduce_channels(x)
        x = self.concat([x, x_skip])
        return self.conv_block(x)


class MlpBlock(keras.Sequential):
    """A multi layer perceptron block."""

    def __init__(
        self,
        n_outputs: int,
        n_features: int,
        n_layers: int,
    ):
        super().__init__()
        for _ in range(n_layers - 1):
            self.add(
                layers.Conv2D(n_features, 1, padding="same", use_bias=False)
            )
            self.add(layers.Activation(keras.activations.relu))
            # self.add(layers.Dropout(0.5)) # added dropout layer here
        self.add(layers.Conv2D(n_outputs, 1, padding="same"))
