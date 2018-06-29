import tensorflow as tf
import numpy as np
a = tf.constant(np.zeros(5), name='a')
b = tf.constant(np.ones(5), name='b')

c = tf.add(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(c))
