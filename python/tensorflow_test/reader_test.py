"""
    读取先前保存的数据
"""

import tensorflow as tf
import numpy as np

# 需要首先重新定义shape和type
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
biases = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

# 不需要定义init
saver = tf.compat.v1.train.Saver()
with tf.compat.v1.Session() as sess:
    saver.restore(sess, 'save_net/save_net.ckpt')
    print('weights', sess.run(W))
    print('biases', sess.run(biases))

