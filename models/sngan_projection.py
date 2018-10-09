from models.discriminator import SNGANDiscriminator
from models.generator import SNGANGenerator
import tensorflow as tf


class SNGANProjection(tf.layers.Layer):
    """SNGAN Main Model
    """
    def __init__(self,
                 channel=64,
                 category=0,
                 z_dim=128,
                 per_gen_train_steps=5,
                 trainable=True,
                 name=None,
                 gen_kwargs={},
                 disc_kwargs={},
                 **kwargs):
        super(SNGANProjection, self).__init__(
            name=name, trainable=trainable, **kwargs)
        self._channel = channel
        self._category = category
        self._z_dim = z_dim
        self._per_gen_train_steps = per_gen_train_steps
        self._gen_kwargs = gen_kwargs
        self._disc_kwargs = disc_kwargs
        self._gen_optimizer = tf.train.AdamOptimizer()
        self._disc_optimizer = tf.train.AdamOptimizer()
        self._layers = []

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        self.discriminator = SNGANDiscriminator(channel=self._channel,
                                                category=self._category,
                                                trainable=self.trainable,
                                                **self._disc_kwargs)

        self.generator = SNGANGenerator(channel=self._channel,
                                        category=self._category,
                                        trainable=self.trainable,
                                        **self._gen_kwargs)

        self._layers += [self.discriminator, self.generator]
        self.built = True

    @property
    def variables(self):
        vars = [self.embed_y]
        for l in self._layers:
            vars += l.variables
        return vars

    def call(self, inputs, labels=None):
        batch_size = inputs.shape[0]
        z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        gens = self.generator(z, labels=labels)  # 本実装ではラベルもサンプリングしている
        disc_true = self.discriminator(inputs, labels=labels)
        disc_fake = self.discriminator(gens, labels=labels)

        # Hinge Loss
        disc_loss = tf.reduce_mean(tf.nn.relu(1 - disc_true))
        disc_loss += tf.reduce_mean(tf.nn.relu(1 + disc_fake))

        gen_loss = -tf.reduce_mean(disc_fake)
        return disc_loss, gen_loss

    def optimize(self, disc_loss, gen_loss, clipped_value=1.0):
        # ジェネレーターはN回に一回のみトレーニングする
        def gen_train(gen_loss):
            grads = self._gen_optimizer.compute_gradients(gen_loss, self.generator.variables)
            clipped_grads = [(tf.clip_by_value(g, -clipped_value, clipped_value), v) for g, v in grads]
            train_op = self._gen_optimizer.apply_gradients(clipped_grads)
            return train_op

        global_steps = tf.train.get_or_create_global_step()
        gen_train_op = tf.cond(global_steps % 5 == 0, gen_train, lambda: tf.no_op())

        with tf.control_dependencies([gen_train_op]):
            grads = self._disc_optimizer.compute_gradients(disc_loss, self.discriminator.variables)
            clipped_grads = [(tf.clip_by_value(g, -clipped_value, clipped_value), v) for g, v in grads]
            train_op = self._disc_optimizer.apply_gradients(clipped_grads)
            return train_op
