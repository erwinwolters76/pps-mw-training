from pathlib import Path

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore


# image size
Y = 272
X = 224


def get_training_dataset(
    training_data_path: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    channels: list[int],
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Get training dataset."""
    with open(training_data_path / "tb1.npy", "rb") as f:
        tb = np.load(f)
    with open(training_data_path / "dbz1.npy", "rb") as f:
        dbz = np.load(f)
    tb = np.moveaxis(tb[:, channels, 0:Y, 0:X], 1, -1).astype(np.float32)
    dbz = np.expand_dims(dbz[:, 0:Y, 0:X], axis=3).astype(np.float32)
    n_samples = tb.shape[0]
    fractions = [train_fraction, validation_fraction, test_fraction]
    limits = np.cumsum([int(f * n_samples) for f in fractions])
    idxs = np.random.permutation(n_samples)
    idxs_train = idxs[0: limits[0]]
    idxs_val = idxs[limits[0]: limits[1]]
    idxs_test = idxs[limits[1]: limits[2]]
    return (
        tf.data.Dataset.from_tensor_slices((tb[idxs_train], dbz[idxs_train])),
        tf.data.Dataset.from_tensor_slices((tb[idxs_val], dbz[idxs_val])),
        tf.data.Dataset.from_tensor_slices((tb[idxs_test], dbz[idxs_test])),
    )
