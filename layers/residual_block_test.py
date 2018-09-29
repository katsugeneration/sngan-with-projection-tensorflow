import numpy as np
import tensorflow as tf
from layers.residual_block import ResidualBlock
from layers.conditional_batch_normalization import ConditionalBatchNormalization


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
        self.assertEqual(rb.conv1_u, None)
        self.assertEqual(rb.conv2_u, None)

    def testBuildAndRunWithUpsampling(self):
        N = 5
        W = 10
        H = 10
        C = 3
        hidden_c = 5
        x = tf.ones((N, H, W, C))
        rb = ResidualBlock(hidden_c=hidden_c, upsampling=True)
        outputs = rb(x)

        self.assertEqual(rb.in_c, C)
        self.assertEqual(rb.out_c, C)
        self.assertEqual(rb.hidden_c, hidden_c)
        self.assertEqual((N, H * 2, W * 2, C), outputs.shape)
        self.assertEqual((rb.ksize, rb.ksize, C, hidden_c), rb.conv1.kernel.shape)
        self.assertEqual((1, 1, C, C), rb.conv_shortcut.kernel.shape)
        self.assertTrue(hasattr(rb, 'bn1'))
        self.assertEqual(rb.conv1_u, None)
        self.assertEqual(rb.conv2_u, None)
        self.assertEqual(rb.conv_shortcut_u, None)

    def testBuildAndRunWithDownsampling(self):
        N = 5
        W = 10
        H = 10
        C = 3
        hidden_c = 5
        x = tf.ones((N, H, W, C))
        rb = ResidualBlock(hidden_c=hidden_c, is_use_bn=False, downsampling=True)
        outputs = rb(x)

        self.assertEqual(rb.in_c, C)
        self.assertEqual(rb.out_c, C)
        self.assertEqual(rb.hidden_c, hidden_c)
        self.assertEqual((N, H / 2, W / 2, C), outputs.shape)
        self.assertEqual((rb.ksize, rb.ksize, C, hidden_c), rb.conv1.kernel.shape)
        self.assertEqual((1, 1, C, C), rb.conv_shortcut.kernel.shape)
        self.assertFalse(hasattr(rb, 'bn1'))

    def testBuildAndRunWithDownsamplingWithSN(self):
        N = 5
        W = 10
        H = 10
        C = 3
        hidden_c = 5
        x = tf.ones((N, H, W, C))
        rb = ResidualBlock(hidden_c=hidden_c, is_use_bn=False, downsampling=True, is_use_sn=True)
        outputs = rb(x)

        self.assertEqual(rb.in_c, C)
        self.assertEqual(rb.out_c, C)
        self.assertEqual(rb.hidden_c, hidden_c)
        self.assertEqual((N, H / 2, W / 2, C), outputs.shape)
        self.assertEqual((rb.ksize, rb.ksize, C, hidden_c), rb.conv1.kernel.shape)
        self.assertEqual((1, 1, C, C), rb.conv_shortcut.kernel.shape)
        self.assertFalse(hasattr(rb, 'bn1'))
        self.assertNotEqual(rb.conv1_u, None)
        self.assertNotEqual(rb.conv2_u, None)
        self.assertNotEqual(rb.conv_shortcut_u, None)

    def testTrain(self):
        N = 5
        W = 10
        H = 10
        C = 3
        hidden_c = 5
        x = tf.ones((N, H, W, C))
        rb = ResidualBlock(hidden_c=hidden_c)

        with tf.GradientTape() as tape:
            outputs = rb(x)
            loss = tf.reduce_mean(tf.square(1 - outputs))

        grads = tape.gradient(loss, rb.variables)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        optimizer.apply_gradients(zip(grads, rb.variables))

        self.assertEqual(type(rb.bn1), tf.layers.BatchNormalization)

    def testTrainCategory(self):
        N = 5
        W = 10
        H = 10
        C = 3
        hidden_c = 5
        x = tf.ones((N, H, W, C))
        y = [[3], [1], [3], [1], [3]]
        rb = ResidualBlock(hidden_c=hidden_c, category=4)

        with tf.GradientTape() as tape:
            outputs = rb(x, labels=y)
            loss = tf.reduce_mean(tf.square(1 - outputs))

        grads = tape.gradient(loss, rb.variables)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        optimizer.apply_gradients(zip(grads, rb.variables))

        self.assertEqual(type(rb.bn1), ConditionalBatchNormalization)


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()
