import numpy as np
import tensorflow as tf
from models.sngan_projection import SNGANProjection


class SNGANProjectionTest(tf.test.TestCase):
    def testInit(self):
        SNGANProjection()

    def testBuildAndRun(self):
        N = 5
        W = 10
        H = 10
        C = 3
        x = tf.ones((N, H, W, C))
        y = [[0]] * N
        sngan = SNGANProjection(category=5)
        disc_loss, gen_loss = sngan(x, labels=y)

        self.assertNotEqual(disc_loss.numpy(), [np.NaN])
        self.assertNotEqual(gen_loss.numpy(), [np.NaN])


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()
