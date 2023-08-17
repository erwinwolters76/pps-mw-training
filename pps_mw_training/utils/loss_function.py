import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore


def quantile_loss(
    n_params: int,
    quantiles: list[float],
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    fill_value: float = -100.0,
) -> tf.Tensor:
    """Quantile loss function handling multiple quantiles and parameters."""
    s = len(quantiles)
    q = tf.constant(np.tile(quantiles, n_params), dtype=tf.float32)
    if n_params == 1:
        e = y_true - y_pred
        e = tf.where(y_true == fill_value, 0., e)
    else:
        e = tf.concat(
            [
                tf.expand_dims(y_true[:, i], 1) - y_pred[:, i * s: (i + 1) * s]
                for i in range(n_params)
            ],
            axis=1
        )
    return tf.reduce_mean(
        tf.maximum(q * e, (q - 1) * e)
    )
