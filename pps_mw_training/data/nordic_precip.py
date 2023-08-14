import numpy as np  # type: ignore

import tensorflow as tf  # type: ignore


AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 128

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
])


def get_training_dataset() -> tf.data.Dataset:
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


def prepare_dataset(
    ds: tf.data.Dataset,
    n_channels: int,
    batch_size: int,
    augment: bool = False,
) -> tf.data.Dataset:
    ds = ds.batch(batch_size)
    if augment:
        ds = ds.map(
            lambda x, y: tf.split(
                data_augmentation(
                    tf.concat([x, y], axis=3),
                    training=True,
                ),
                [n_channels, 1],
                axis=3
            ),
            num_parallel_calls=AUTOTUNE,
        )
    return ds.prefetch(buffer_size=AUTOTUNE)
