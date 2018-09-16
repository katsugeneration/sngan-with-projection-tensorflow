import tensorflow as tf
from layers.conditional_batch_normalization import ConditionalBatchNormalization


class ResidualBlock(tf.layers.Layer):
    '''Residual Block Layer
    '''

    def __init__(self,
                 out_c=None,
                 hidden_c=None,
                 ksize=3,
                 stride=1,
                 activation=tf.nn.relu,
                 is_use_bn=True,
                 upsampling=False,
                 downsampling=False,
                 category=0,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(ResidualBlock, self).__init__(
            name=name, trainable=trainable, **kwargs)
        self.out_c = out_c
        self.hidden_c = hidden_c
        self.ksize = ksize
        self.stride = stride
        self.activation = activation
        self.is_use_bn = is_use_bn
        self.upsampling = upsampling
        self.downsampling = downsampling
        self.category = category
        self._layers = []

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        self.in_c = input_shape[-1]
        if self.out_c is None:
            self.out_c = self.in_c
        if self.hidden_c is None:
            self.hidden_c = self.in_c
        self.is_shortcut_learn = (self.in_c != self.out_c) or self.upsampling or self.downsampling

        self.conv1 = tf.layers.Conv2D(self.hidden_c,
                                      self.ksize,
                                      strides=(self.stride, self.stride),
                                      padding='SAME',
                                      use_bias=False,
                                      activation=None,
                                      kernel_initializer=tf.initializers.random_uniform())
        self._layers.append(self.conv1)

        self.conv2 = tf.layers.Conv2D(self.out_c,
                                      self.ksize,
                                      strides=(self.stride, self.stride),
                                      padding='SAME',
                                      use_bias=False,
                                      activation=None,
                                      kernel_initializer=tf.initializers.random_uniform())
        self._layers.append(self.conv2)

        if self.is_use_bn:
            if self.category:
                self.bn1 = ConditionalBatchNormalization(self.category)
                self._layers.append(self.bn1)
                self.bn2 = ConditionalBatchNormalization(self.category)
                self._layers.append(self.bn2)
            else:
                self.bn1 = tf.layers.BatchNormalization()
                self._layers.append(self.bn1)
                self.bn2 = tf.layers.BatchNormalization()
                self._layers.append(self.bn2)

        if self.is_shortcut_learn:
            self.conv_shortcut = tf.layers.Conv2D(self.out_c,
                                                  1,
                                                  strides=(1, 1),
                                                  padding='SAME',
                                                  use_bias=False,
                                                  activation=None,
                                                  kernel_initializer=tf.initializers.random_uniform())
            self._layers.append(self.conv_shortcut)

    @property
    def variables(self):
        vars = []
        for l in self._layers:
            vars += l.variables
        return vars

    def _upsample(self, var):
        height = var.shape[1]
        width = var.shape[2]
        return tf.image.resize_images(var,
                                      (height * 2, width * 2),
                                      method=tf.image.ResizeMethod.BILINEAR)

    def _downsample(self, var):
        return tf.layers.average_pooling2d(var, (2, 2), (2, 2), padding='SAME')

    def call(self, inputs, labels=None):
        out = inputs
        if self.is_use_bn:
            out = self.bn1(out) if labels is None else self.bn1(out, labels=labels)
        out = self.activation(out)
        if self.upsampling:
            out = self._upsample(out)
        out = self.conv1(out)
        if self.is_use_bn:
            out = self.bn2(out) if labels is None else self.bn2(out, labels=labels)
        out = self.activation(out)
        out = self.conv2(out)
        if self.downsampling:
            out = self._downsample(out)

        if self.is_shortcut_learn:
            if self.upsampling:
                x = self.conv_shortcut(self._upsample(inputs))
            elif self.downsampling:
                x = self._downsample(self.conv_shortcut(inputs))
            else:
                x = self.conv_shortcut(inputs)
        else:
            x = inputs
        return out + x

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        height = input_shape[1]
        width = input_shape[2]
        if self.upsampling:
            height *= 2
            width *= 2
        return tf.TensorShape(
          [input_shape[0], height, width, self.out_c])
