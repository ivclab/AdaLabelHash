import tensorflow as tf
import keras.backend as K


def tf_repeat(tensor, repeats):
    """
    Args:
        input: A Tensor. 1-D or higher.
        repeats: A list. Number of repeat for each dimension, length must be the
                 same as the number of dimensions in input

    Returns:
        A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats

    Provided by qianyizhang in https://github.com/tensorflow/tensorflow/issues/8246
    """
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor


def construct_triplet_loss(num_classes=10, k=1.0):
    """ Triplet loss
    # Arguments:
        num_classes: Number of classes
        k: Margin value
    """
    def triplet_loss(y_true, y_pred):
        mask_0 = K.cast(K.equal(y_true, 0), K.floatx())
        mask_1 = K.cast(K.equal(y_true, 1), K.floatx())

        values, _ = tf.nn.top_k(y_pred * mask_0, 1)
        values = tf.tile(values, [1, num_classes])

        y_pred = y_pred * mask_1
        values = values * mask_1
        pt = mask_1*k - y_pred + values
        pt = tf.where(tf.greater(pt, 0), pt, mask_1*0)
        return K.sum(pt) / K.sum(mask_1)
    return triplet_loss
