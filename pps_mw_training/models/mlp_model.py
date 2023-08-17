from tensorflow.keras import Input, Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore


class MlpModel(Sequential):
    """Multi layer perceptron model."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden_layers: int,
        n_neurons_per_layer: int,
        activation: str,
    ):
        super().__init__()
        self.add(Input(shape=(n_inputs,)))
        for _ in range(n_hidden_layers):
            self.add(Dense(n_neurons_per_layer, activation=activation))
        self.add(Dense(n_outputs, activation="linear"))
