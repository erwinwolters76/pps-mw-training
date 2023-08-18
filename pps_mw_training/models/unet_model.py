import tensorflow as tf  # type: ignore
from tensorflow import keras

from pps_mw_training.utils.blocks import (
    ConvolutionBlock,
    DownsamplingBlock,
    MlpBlock,
    UpsamplingBlock,
)
from pps_mw_training.utils.layers import UpSampling2D


class UnetModel(keras.Model):
    """U-Net convolutional neural network model."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_unet_base: int,
        n_blocks: int,
        n_features: int,
        n_layers: int,
    ):
        super().__init__()
        self.input_block = ConvolutionBlock(n_inputs, n_unet_base)
        self.down_sampling_blocks = [
            DownsamplingBlock(
                n_unet_base * 2 ** i,
                n_unet_base * 2 ** (i + 1),
            ) for i in range(n_blocks)
        ]
        self.up_sampling_blocks = [
            UpsamplingBlock(
                n_unet_base * 2 ** (i + 1),
                n_unet_base * 2 ** i,
            ) for i in range(n_blocks - 1, -1, -1)
        ]
        self.up_sampling_layer = UpSampling2D()
        self.output_block = MlpBlock(n_outputs, n_features, n_layers)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        xs = []
        x = self.input_block(inputs)
        xs.append(x)
        for down_block in self.down_sampling_blocks:
            x = down_block(x)
            xs.append(x)
        for idx, up_block in enumerate(self.up_sampling_blocks):
            x = up_block([x, xs[-2 - idx]])
        x = self.up_sampling_layer(x)
        return self.output_block(x)

    def build_graph(self, image_size: int, n_inputs: int):
        x = keras.Input(shape=(image_size, image_size, n_inputs))
        return keras.Model(inputs=[x], outputs=self.call(x))
