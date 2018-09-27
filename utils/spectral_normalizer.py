import tensorflow as tf


def spectral_normalizer(W: tf.Variable):
    scope = "/".join(W.name.split("/")[:-1])
    with tf.variable_scope(scope):
        u = tf.get_variable(
                        name="u",
                        shape=(1, W.shape[0]),
                        initializer=tf.initializers.random_normal())
    v = tf.nn.l2_normalize(tf.matmul(u, W))
    _u = tf.nn.l2_normalize(tf.matmul(v, W, transpose_b=True))
    sigma = tf.matmul(tf.matmul(_u, W), v, transpose_b=True)
    sigma = tf.reduce_sum(sigma)
    u = u.assign(_u)
    return W / sigma
