import numpy as np
import tensorflow as tf
from residual_block import ResidualBlock

tf.enable_eager_execution()


class ResidualBlockTest(tf.test.TestCase):
    def testInit(self):
        ResidualBlock()

    def testBuildAndRun(self):
        N = 5
        W = 10
        H = 10
        C = 3
        hidden_c = 5
        x = tf.ones((N, H, W, C))
        rb = ResidualBlock(hidden_c=hidden_c)
        outputs = rb(x)

        self.assertEqual(rb.in_c, C)
        self.assertEqual(rb.out_c, C)
        self.assertEqual(rb.hidden_c, hidden_c)
        self.assertEqual((N, H, W, C), outputs.shape)
        self.assertEqual((rb.ksize, rb.ksize, C, hidden_c), rb.conv1.kernel.shape)


tf.test.main()
