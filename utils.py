"""Generic utility functions used in different sessions."""

import tensorflow as tf


def tf_unique_name(name):
    idx = tf.keras.backend.get_uid(name)
    return f"{name}_{idx}"
