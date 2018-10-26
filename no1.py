import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;

# tensor = tf.constant(1.0, verify_shape=True)
# tensor = [1, 2, 3, 4, 5, 6]
# tensor = [1.0, 5.5, 9.0, 2.0, 6.1, 8.5, 7.9, 3.7, 4.0]
# b = tf.nn.dropout(tensor, 0.5, noise_shape=None, seed=None)

# c = tf.reshape(tensor, shape=[-1])
sess = tf.Session()

b = tf.constant(1, shape=[5, 5])
a = sess.run(b)
print(a)




