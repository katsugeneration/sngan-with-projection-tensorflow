import nose
import tensorflow as tf

tf.enable_eager_execution()

config = nose.config.Config()
config.configure(argv=["--exclude", "test.py"])
result = nose.run(config=config)
