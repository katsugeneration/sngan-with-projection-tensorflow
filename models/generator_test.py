import numpy as np
import tensorflow as tf
from models.generator import SNGANGenerator


class SNGANGeneratorTest(tf.test.TestCase):
    def testInit(self):
        SNGANGenerator()

    def testBuildAndRun(self):
        N = 5
        z_size = 10
        z = tf.initializers.random_normal()((N, z_size))
        sng = SNGANGenerator(category=5)
        y = [[0]] * N
        outputs = sng(z, labels=y)
        o_H = sng._bottom_w * (2 ** 5)

        self.assertEqual((N, o_H, o_H, 3), outputs.shape)

    def testTrain(self):
        N = 5
        z_size = 10
        z = tf.initializers.random_normal()((N, z_size))
        sng = SNGANGenerator(category=5)
        y = [[0]] * N

        with tf.GradientTape() as tape:
            outputs = sng(z, labels=y)
            loss = tf.reduce_mean(tf.square(1 - outputs))

        grads = tape.gradient(loss, sng.variables)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        optimizer.apply_gradients(zip(grads, sng.variables))
