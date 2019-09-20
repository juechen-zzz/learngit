"""
    可以看到训练过程，可视化
"""

import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(1)
np.random.seed(1)

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

with tf.compat.v1.variable_scope('Inputs'):
    tf_x = tf.compat.v1.placeholder(tf.float32, x.shape, name='x')
    tf_y = tf.compat.v1.placeholder(tf.float32, y.shape, name='y')

with tf.variable_scope('Net'):
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden_layer')
    output = tf.layers.dense(l1, 1, name='output_layer')

    # add to histogram summary（画隐藏和输出层的结果图）
    tf.compat.v1.summary.histogram('h_out', l1)
    tf.compat.v1.summary.histogram('pred', output)

loss = tf.compat.v1.losses.mean_squared_error(tf_y, output, scope='loss')
train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

tf.compat.v1.summary.scalar('loss', loss)     # add loss to scalar summary（画损失函数的迭代图）

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

writer = tf.compat.v1.summary.FileWriter('./log', sess.graph)     # write to file（写文件）
merge_op = tf.compat.v1.summary.merge_all()                       # operation to merge all summary（整合所有图）

for step in range(100):
    # train and net output
    _, result = sess.run([train_op, merge_op], {tf_x: x, tf_y: y})
    writer.add_summary(result, step)                              # 同训练一起迭代

# Lastly, in your terminal or CMD, type this :
# $ tensorboard --logdir path/to/log
# open you google chrome, type the link shown on your terminal or CMD. (something like this: http://localhost:6006)