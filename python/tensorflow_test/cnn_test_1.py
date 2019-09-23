"""
    CNN实现
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    # 输入v_xs进行预测
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # 计算是否相等
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    # 转换形式后求均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 返回结果
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    # 从截断的正态分布中输出随机值，stddev是指的标注差
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides含义为[batch, height, weight, channel]，首尾为1指不跳过任何一个样本和颜色通道，每一个样本及通道都会计算
    return tf.compat.v1.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2(x):
    # ksize 池化窗口的大小，是一个四维向量，[batch, height, weight, channel]
    return tf.compat.v1.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义数据
xs = tf.compat.v1.placeholder(tf.float32, [None, 784])
ys = tf.compat.v1.placeholder(tf.float32, [None, 10])
keep_prob = tf.compat.v1.placeholder(tf.float32)
x_image = tf.compat.v1.reshape(xs, [-1, 28, 28, 1])     # 定义有n个28*28、通道为1(黑白)的图，n根据传入的参数自己匹配
print(x_image.shape)        # [n_samples, 28, 28, 1]

# 定义第1层(卷积层)
W_conv1 = weight_variable([5, 5, 1, 32])                # patch 5*5， in_size：1，out_size：32
b_conv1 = bias_variable([32])
h_conv1 = tf.compat.v1.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2(h_conv1)

# 定义第2层(卷积层)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.compat.v1.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2(h_conv2)

# 定义第3层(全连接层)
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.compat.v1.reshape(h_pool2, [-1, 7*7*64])      # # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_fc1 = tf.compat.v1.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.compat.v1.nn.dropout(h_fc1, keep_prob)

# 定义第4层(全连接层)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.compat.v1.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.math.log(prediction), reduction_indices=[1]))
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)         # 0.0001


sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
