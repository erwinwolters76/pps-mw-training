import tensorflow as tf  # type: ignore


AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 64  # 128


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


@tf.function
def set_missing_data(
    x: tf.Tensor,
    missing_fraction: float,
    fill_value: float,
) -> tf.Tensor:
    """Set a fraction of the data to a given fill value."""
    return tf.where(
        tf.math.greater(
            tf.random.uniform(
                shape=(tf.shape(x)[0], tf.shape(x)[1]),
                minval=0,
                maxval=1,
            ),
            missing_fraction
        ),
        x,
        fill_value,
    )
