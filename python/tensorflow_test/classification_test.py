"""
    实现分类的小实验
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../pytorch_test/mnist', one_hot=True)

def add_layer(inputs, in_size, out_size, activition_function=None):
    Weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activition_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activition_function(Wx_plus_b)

    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))    # argmax返回最大数值所在下标，1表示按行比较返回
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast数据类型转换
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

xs = tf.compat.v1.placeholder(tf.float32, [None, 784])    # 不规定一次输入的图片张数，但是规定一次784个值，就对应一张图的像素点
ys = tf.compat.v1.placeholder(tf.float32, [None, 10])     # 对应0到9

prediction = add_layer(xs, 784, 10, activition_function=tf.nn.softmax)

# 损失函数，交叉熵
# reduction_indices指的是用求和方法压缩第几维，为1时是将第一维（列）压缩，就是每一行求和
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images. mnist.test.labels))

