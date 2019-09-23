"""
    CNN实现
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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

def bias_weight(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides含义为[batch, height, weight, channel]，首尾为1指不跳过任何一个样本和颜色通道，每一个样本及通道都会计算
    return tf.compat.v1.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2(x):
    # ksize 池化窗口的大小，是一个四维向量，[batch, height, weight, channel]
    return tf.compat.v1.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
