import tensorflow as tf
from layers.residual_block import ResidualBlock
from utils.spectral_normalizer import spectral_normalizer


class SNGANDiscriminator(tf.layers.Layer):
    """SNGAN Discriminator
    """
    def __init__(self,
                 channel=64,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(SNGANDiscriminator, self).__init__(
            name=name, trainable=trainable, **kwargs)
        self._channel = channel
        self._activation = activation
        self._category = category
        self._layers = []

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        self.block1 = ResidualBlock(out_c=self._channel,
                                    activation=self._activation,
                                    is_use_bn=False)
        self.block2 = ResidualBlock(out_c=self._channel * 2,
                                    activation=self._activation,
                                    is_use_bn=False,
                                    downsampling=True)
        self.block3 = ResidualBlock(out_c=self._channel * 4,
                                    activation=self._activation,
                                    is_use_bn=False,
                                    downsampling=True)
        self.block4 = ResidualBlock(out_c=self._channel * 8,
                                    activation=self._activation,
                                    is_use_bn=False,
                                    downsampling=True)
        self.block5 = ResidualBlock(out_c=self._channel * 16,
                                    activation=self._activation,
                                    is_use_bn=False,
                                    downsampling=True)
        self.block6 = ResidualBlock(out_c=self._channel * 16,
                                    activation=self._activation,
                                    is_use_bn=False,
                                    downsampling=True)

        self.dense = tf.layers.Dense(1,
                                     use_bias=False,
                                     activation=None,
                                     kernel_initializer=tf.initializers.random_normal())
        self.dense_u = None

        self._layers += [self.block1,
                         self.block2,
                         self.block3,
                         self.block4,
                         self.block5,
                         self.block6,
                         self.dense]

        if self._category != 0:
            self.embed_y = self.add_weight(
                                'embeddings',
                                shape=[self._category, self._channel * 16],
                                initializer=tf.initializers.random_normal(),
                                regularizer=None,
                                constraint=None,
                                trainable=True)
            self.embed_u = tf.get_variable(
                            name="u",
                            shape=(1, self._category),
                            initializer=tf.initializers.random_normal(),
                            trainable=False)

    @property
    def variables(self):
        vars = [self.embed_y]
        for l in self._layers:
            vars += l.variables
        return vars

    def call(self, inputs, labels=None):
        out = inputs

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self._activation(out)
        out = tf.reduce_sum(out, axis=(1, 2))
        h = out

        if self.dense_u is None:
            with tf.variable_scope("dense"):
                self.dense_u = tf.get_variable(
                                    name="u",
                                    shape=(1, out.shape[-1]),
                                    initializer=tf.initializers.random_normal(),
                                    trainable=False)
        if not self.dense.built:
            self.dense.build(out.shape)
        with spectral_normalizer(self.dense.kernel, self.dense_u):
            out = self.dense(out)

        if labels is not None:
            with spectral_normalizer(self.embed_y, self.embed_u):
                w_y = tf.nn.embedding_lookup(self.embed_y, labels)
                w_y = tf.reshape(w_y, (-1, self._channel * 16))
                out += tf.reduce_sum(w_y * h, axis=1, keepdims=True)

        return out
