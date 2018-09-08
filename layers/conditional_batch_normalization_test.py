import numpy as np
import tensorflow as tf
from conditional_batch_normalization import ConditionalBatchNormalization

tf.enable_eager_execution()


class ConditionalBatchNormalizationTest(tf.test.TestCase):
    def testInit(self):
        ConditionalBatchNormalization()

    def testBuildAndRun(self):
        cbn = ConditionalBatchNormalization()
        N = 5
        W = 10
        H = 10
        C = 3
        x = tf.ones((N, H, W, C))
        outputs = cbn(x)
        self.assertEqual(x.shape, outputs.shape)
        self.assertEqual((C, ), cbn.gamma.shape)
        self.assertEqual((C, ), cbn.beta.shape)
        self.assertEqual((C, ), cbn.moving_mean.shape)
        self.assertEqual((C, ), cbn.moving_variance.shape)
        self.assertAllEqual([0, 0, 0], cbn.moving_mean.numpy())
        self.assertAllEqual([1, 1, 1], cbn.moving_variance.numpy())

    def testTrain(self):
        cbn = ConditionalBatchNormalization(momentum=0.99)
        N = 5
        W = 10
        H = 10
        C = 3
        x = tf.ones((N, H, W, C))

        with tf.GradientTape() as tape:
            outputs = cbn(x, training=True)
            loss = tf.reduce_mean(tf.square(1 - outputs))

        print(loss)
        grads = tape.gradient(loss, cbn.variables)
        print(grads)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        optimizer.apply_gradients(zip(grads, cbn.variables))

        self.assertAllEqual(np.array([0.01, 0.01, 0.01], dtype=np.float32), cbn.moving_mean.numpy())
        self.assertAllEqual(np.array([0.99, 0.99, 0.99], dtype=np.float32), cbn.moving_variance.numpy())
        self.assertAllEqual(np.array([1, 1, 1], dtype=np.float32), cbn.gamma.numpy())
        self.assertAllEqual(-grads[1] * 0.001, cbn.beta.numpy())


tf.test.main()
