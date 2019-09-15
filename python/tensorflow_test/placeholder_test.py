"""
    placeholder 放传入样本值，与feed_dict绑定
"""

import tensorflow as tf

input_1 = tf.placeholder(tf.float32)          # 给定type，一般默认float32
input_2 = tf.placeholder(tf.float32)

# multiply实现元素乘法，即矩阵中对应元素相乘，matmul实现的是矩阵乘法
output = tf.multiply(input_1, input_2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input_1: [7.], input_2: [2.]}))       # feed_dict用来传入值
