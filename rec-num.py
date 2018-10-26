import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# print('Training data size:', mnist.train.num_examples)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])     # 输入数据的地方

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)         # y=softmax(Wx+b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))     # 损失函数

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)      # 优化并训练

tf.global_variables_initializer().run()

for i in range(1000):                         # 迭代训练
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_predition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))




