import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore


def get_training_dataset(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tf.data.Dataset:
    """Get training dataset, experimental code so far."""
    with open('tb1.npy', 'rb') as f:
        tb = np.load(f)
        idx = [14, 15, 16, 17, 18]
        tb = tb[:, idx, 0:272, 0:224]
    with open('dbz1.npy', 'rb') as f:
        dbz = np.load(f)
        dbz = dbz[:, 0:272, 0:224]
    return (
        tf.data.Dataset.from_tensor_slices(
            (
                np.moveaxis(tb, 1, -1).astype(np.float32),
                np.expand_dims(dbz, axis=3).astype(np.float32),
            )
        ),
        tf.data.Dataset.from_tensor_slices(
            (
                np.moveaxis(tb, 1, -1).astype(np.float32),
                np.expand_dims(dbz, axis=3).astype(np.float32),
            )
        ),
        tf.data.Dataset.from_tensor_slices(
            (
                np.moveaxis(tb, 1, -1).astype(np.float32),
                np.expand_dims(dbz, axis=3).astype(np.float32),
            )
        ),
    )
