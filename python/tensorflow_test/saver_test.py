"""
    保存数据
"""

import tensorflow as tf

# 保存到文件
# 记住保存为相同的dtype和shape
W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
biases = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

init = tf.compat.v1.global_variables_initializer()

saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, '/Users/nihaopeng/个人/Git/learngit/python/tensorflow_test/save_net/save_net.ckpt')
    print('Save to path:', save_path)