import tensorflow as tf  # type: ignore


@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
))
def random_crop_and_flip(
    x,
    y,
    image_size,
):
    """Apply random crop and flip."""
    x, y = random_crop(x, y, image_size)
    x, y = random_flip(x, y)
    return x, y


@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.float32),
))
def random_flip(
    x,
    y,
):
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


@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
))
def random_crop(
    x,
    y,
    image_size,
):
    """Random crop of data."""
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    n1 = tf.cast(y_shape[1] / x_shape[1], tf.int32)
    n2 = tf.cast(y_shape[2] / x_shape[2], tf.int32)
    s1 = tf.random.uniform(
        (),
        minval=0,
        maxval=x_shape[1] - image_size,
        dtype=tf.dtypes.int32,
    )
    s2 = tf.random.uniform(
        (),
        minval=0,
        maxval=x_shape[2] - image_size,
        dtype=tf.dtypes.int32,
    )
    return (
        x[:, s1: s1 + image_size, s2: s2 + image_size, :],
        y[
            :,
            s1 * n1: (s1 + image_size) * n1,
            s2 * n2: (s2 + image_size) * n2, :,
        ],
    )


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
