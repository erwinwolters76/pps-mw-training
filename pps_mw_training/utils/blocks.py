import tensorflow as tf  # type: ignore
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore


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
        self.block.add(SymmetricPadding(1))
        self.block.add(layers.Conv2D(channels_out, 3, padding="valid"))
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())

    def call(self, input):
        x = input
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
    to increase the input size. This is followed by a 1x1 convolution to
    reduce the number of channels, concatenation of the skip inputs
    from the corresponding downsampling layer and a convolution block.
    """

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.upsample = layers.UpSampling2D(
            size=(2, 2),
            interpolation="bilinear",
        )
        input_shape = (None, None, channels_in)
        self.reduce = layers.Conv2D(
            channels_in // 2, 1, padding="same", input_shape=input_shape
        )
        self.concat = layers.Concatenate()
        self.conv_block = ConvolutionBlock(channels_in, channels_out)

    def call(self, inputs):
        x, x_skip = inputs
        x_up = self.reduce(self.upsample(x))
        x = self.concat([x_up, x_skip])
        return self.conv_block(x)


class MLP(keras.Sequential):
    """A multi layer perceptron block."""

    def __init__(
        self,
        n_outputs: int,
        n_features: int = 128,
        n_layers: int = 4,
    ):
        super().__init__()
        for _ in range(n_layers - 1):
            self.add(
                layers.Conv2D(n_features, 1, padding="same", use_bias=False)
            )
            self.add(layers.Activation(keras.activations.relu))
        self.add(layers.Conv2D(n_outputs, 1, padding="same"))


class SymmetricPadding(layers.Layer):
    """Block for symmetric padding."""

    def __init__(self, amount: int):
        super().__init__()
        self.paddings = tf.constant(
            [[0, 0], [amount, amount], [amount, amount], [0, 0]]
        )

    def call(self, input):
        return tf.pad(input, self.paddings, "SYMMETRIC")
