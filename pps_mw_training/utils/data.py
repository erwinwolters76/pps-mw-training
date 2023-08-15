import tensorflow as tf  # type: ignore


AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 128


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
])


def prepare_dataset(
    ds: tf.data.Dataset,
    n_channels: int,
    batch_size: int,
    augment: bool = False,
) -> tf.data.Dataset:
    """Prepare dataset."""
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
