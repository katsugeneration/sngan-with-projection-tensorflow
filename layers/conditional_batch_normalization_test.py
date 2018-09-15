import numpy as np
import tensorflow as tf
from layers.conditional_batch_normalization import ConditionalBatchNormalization

tf.enable_eager_execution()


class ConditionalBatchNormalizationTest(tf.test.TestCase):
    def testInit(self):
        ConditionalBatchNormalization(4)

    def testBuildAndRun(self):
        CAT = 4
        cbn = ConditionalBatchNormalization(CAT)
        N = 5
        W = 10
        H = 10
        C = 3
        x = tf.ones((N, H, W, C))
        y = [[0]] * N
        outputs = cbn(x, labels=y)
        self.assertEqual(x.shape, outputs.shape)
        self.assertEqual((CAT, C, ), cbn.gamma.shape)
        self.assertEqual((CAT, C, ), cbn.beta.shape)
        self.assertEqual((C, ), cbn.moving_mean.shape)
        self.assertEqual((C, ), cbn.moving_variance.shape)
        self.assertAllEqual([0, 0, 0], cbn.moving_mean.numpy())
        self.assertAllEqual([1, 1, 1], cbn.moving_variance.numpy())

    def testTrain(self):
        CAT = 4
        cbn = ConditionalBatchNormalization(CAT, momentum=0.99)
        N = 5
        W = 10
        H = 10
        C = 3
        x = tf.ones((N, H, W, C))
        y = [[0]] * N

        with tf.GradientTape() as tape:
            outputs = cbn(x, labels=y, training=True)
            loss = tf.reduce_mean(tf.square(1 - outputs))

        grads = tape.gradient(loss, cbn.variables)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        optimizer.apply_gradients(zip(grads, cbn.variables))

        self.assertAllEqual(np.array([0.01, 0.01, 0.01], dtype=np.float32), cbn.moving_mean.numpy())
        self.assertAllEqual(np.array([0.99, 0.99, 0.99], dtype=np.float32), cbn.moving_variance.numpy())
        self.assertAllEqual(np.array([1, 1, 1], dtype=np.float32), cbn.gamma.numpy()[0])
        self.assertAllEqual(np.sum(-grads[1].values, axis=0) * 0.001, cbn.beta[0].numpy())
        self.assertAllEqual(np.array([0, 0, 0], dtype=np.float32), cbn.beta[1].numpy())

    def testTrainCategory(self):
        CAT = 4
        cbn = ConditionalBatchNormalization(CAT, momentum=0.99)
        N = 5
        W = 10
        H = 10
        C = 3
        x = tf.ones((N, H, W, C))
        y = [[3], [1], [3], [1], [3]]

        with tf.GradientTape() as tape:
            outputs = cbn(x, labels=y, training=True)
            loss = tf.reduce_mean(tf.square(1 - outputs))

        grads = tape.gradient(loss, cbn.variables)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        optimizer.apply_gradients(zip(grads, cbn.variables))

        self.assertAllEqual(np.array([0.01, 0.01, 0.01], dtype=np.float32), cbn.moving_mean.numpy())
        self.assertAllEqual(np.array([0.99, 0.99, 0.99], dtype=np.float32), cbn.moving_variance.numpy())
        self.assertAllEqual(np.array([1, 1, 1], dtype=np.float32), cbn.gamma.numpy()[0])
        self.assertAllEqual(np.array([0, 0, 0], dtype=np.float32), cbn.beta[0].numpy())
        self.assertAllEqual(np.array([0, 0, 0], dtype=np.float32), cbn.beta[2].numpy())
        self.assertAllEqual(-(grads[1].values[1] + grads[1].values[3]) * 0.001, cbn.beta[1].numpy())
        self.assertAllEqual(-(grads[1].values[0] + grads[1].values[2] + grads[1].values[4]) * 0.001, cbn.beta[3].numpy())


tf.test.main()
