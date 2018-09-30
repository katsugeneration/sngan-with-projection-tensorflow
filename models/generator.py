import tensorflow as tf
from layers.residual_block import ResidualBlock


class SNGANGenerator(tf.layers.Layer):
    """SNGAN Generator
    """
    def __init__(self,
                 channel=64,
                 bottom_w=4,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(SNGANGenerator, self).__init__(
            name=name, trainable=trainable, **kwargs)
        self._channel = channel
        self._bottom_w = bottom_w
        self._activation = activation
        self._category = category
        self._layers = []

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        
        self.dense = tf.layers.Dense((self._bottom_w ** 2) * self._channel * 16,
                                     use_bias=False,
                                     activation=None,
                                     kernel_initializer=tf.initializers.random_normal())
        self.block1 = ResidualBlock(out_c=self._channel * 16,
                                    activation=self._activation,
                                    category=self._category,
                                    upsampling=True)
        self.block2 = ResidualBlock(out_c=self._channel * 8,
                                    activation=self._activation,
                                    category=self._category,
                                    upsampling=True)
        self.block3 = ResidualBlock(out_c=self._channel * 4,
                                    activation=self._activation,
                                    category=self._category,
                                    upsampling=True)
        self.block4 = ResidualBlock(out_c=self._channel * 2,
                                    activation=self._activation,
                                    category=self._category,
                                    upsampling=True)
        self.block5 = ResidualBlock(out_c=self._channel,
                                    activation=self._activation,
                                    category=self._category,
                                    upsampling=True)
        self.bn = tf.layers.BatchNormalization()
        self.conv = tf.layers.Conv2D(3,
                                     3,
                                     padding='SAME',
                                     use_bias=False,
                                     activation=None,
                                     kernel_initializer=tf.initializers.random_normal())

        self._layers += [self.dense,
                         self.block1,
                         self.block2,
                         self.block3,
                         self.block4,
                         self.block5,
                         self.bn,
                         self.conv]

    @property
    def variables(self):
        vars = []
        for l in self._layers:
            vars += l.variables
        return vars

    def call(self, inputs, labels=None):
        input_shape = inputs.shape
        batch_size = input_shape[0]

        out = self.dense(inputs)
        out = tf.reshape(out, (batch_size, self._bottom_w, self._bottom_w, -1))
        out = self.block1(out, labels=labels)
        out = self.block2(out, labels=labels)
        out = self.block3(out, labels=labels)
        out = self.block4(out, labels=labels)
        out = self.block5(out, labels=labels)
        out = self.bn(out)
        out = self._activation(out)
        out = self.conv(out)
        out = tf.nn.tanh(out)

        return out
