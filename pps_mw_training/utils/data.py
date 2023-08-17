import tensorflow as tf  # type: ignore


@tf.function
def random_crop_and_flip(
    x: tf.Tensor,
    y: tf.Tensor,
    image_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply random crop and flip."""
    x, y = random_crop(x, y, image_size)
    x, y = random_flip(x, y)
    return x, y


@tf.function
def random_flip(
    x: tf.Tensor,
    y: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random flip of data."""
    horizontal_flip = tf.random.uniform(()) > 0.5
    vertical_flip = tf.random.uniform(()) > 0.5
    if horizontal_flip:
        x = tf.reverse(x, [2])
        y = tf.reverse(y, [2])
    if vertical_flip:
        x = tf.reverse(x, [1])
        y = tf.reverse(y, [1])
    return x, y


@tf.function
def random_crop(
    x: tf.Tensor,
    y: tf.Tensor,
    image_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random crop of data."""
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    n1 = int(y_shape[1] / x_shape[1])
    n2 = int(y_shape[2] / x_shape[2])
    s1 = int(
        tf.random.uniform(
            (),
            minval=0,
            maxval=x_shape[1] - image_size,
            dtype=tf.dtypes.int32,
        )
    )
    s2 = int(
        tf.random.uniform(
            (),
            minval=0,
            maxval=x_shape[2] - image_size,
            dtype=tf.dtypes.int32,
        )
    )
    x = x[:, s1: s1 + image_size, s2: s2 + image_size, :]
    y = y[:, s1 * n1: (s1 + image_size) * n1,  s2 * n2: (s2 + image_size) * n2]
    return x, y


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
