import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore


AUTOTUNE = tf.data.AUTOTUNE


def random_crop_and_flip(
    images: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Apply random crop and flip."""
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomCrop(image_size, image_size),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    ])
    ds = ds.batch(batch_size)
    axis = 3
    ds = ds.map(
        lambda x, y: tf.split(
            augmentation(
                tf.concat([x, y], axis=axis),
                training=True,
            ),
            [images.shape[axis], 1],
            axis=axis,
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
