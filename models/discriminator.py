import tensorflow as tf
from layers.residual_block import ResidualBlock


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
                                    activation=self._activation)
        self.block2 = ResidualBlock(out_c=self._channel * 2,
                                    activation=self._activation,
                                    downsampling=True)
        self.block3 = ResidualBlock(out_c=self._channel * 4,
                                    activation=self._activation,
                                    downsampling=True)
        self.block4 = ResidualBlock(out_c=self._channel * 8,
                                    activation=self._activation,
                                    downsampling=True)
        self.block5 = ResidualBlock(out_c=self._channel * 16,
                                    activation=self._activation,
                                    downsampling=True)
        self.block6 = ResidualBlock(out_c=self._channel * 16,
                                    activation=self._activation,
                                    downsampling=True)

        self.dense = tf.layers.Dense(1,
                                     use_bias=False,
                                     activation=None,
                                     kernel_initializer=tf.initializers.random_uniform())

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
                                initializer=tf.initializers.random_uniform(),
                                regularizer=None,
                                constraint=None,
                                trainable=True)
            tf.keras.layers.Embedding

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
        out = self.dense(out)
        if labels is not None:
            w_y = tf.nn.embedding_lookup(self.embed_y, labels)
            w_y = tf.reshape(w_y, (-1, self._channel * 16))
            out += tf.reduce_sum(w_y * h, axis=1, keepdims=True)

        return out
