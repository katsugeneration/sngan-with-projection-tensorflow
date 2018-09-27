import numpy as np
import tensorflow as tf
from models.discriminator import SNGANDiscriminator


class SNGANGeneratorTest(tf.test.TestCase):
    def testInit(self):
        SNGANDiscriminator()

    def testBuildAndRun(self):
        N = 5
        H = 64
        W = 64
        C = 3
        x = tf.ones((N, H, W, C))
        snd = SNGANDiscriminator(category=5)
        y = [[0]] * N
        outputs = snd(x, labels=y)

        self.assertEqual((N, 1), outputs.shape)

    def testTrain(self):
        N = 5
        H = 64
        W = 64
        C = 3
        x = tf.ones((N, H, W, C))
        snd = SNGANDiscriminator(category=5)
        y = [[0]] * N

        with tf.GradientTape() as tape:
            outputs = snd(x, labels=y)
            loss = tf.reduce_mean(tf.square(1 - outputs))

        grads = tape.gradient(loss, snd.variables)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        optimizer.apply_gradients(zip(grads, snd.variables))


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()