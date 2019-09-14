"""
    tensorflow 处理数据的基本结构
"""

import tensorflow as tf
import numpy as np


# create data
x_data = np.random.rand(100).astype(np.float32)             # 强制类型转换
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start  ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))    # 1维，初始值-1到1
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)          # 选择灰度下降，0.5是学习效率
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()                    # 以上是建立，现在是初始化结构，使之活动起来
### create tensorflow structure start  ###

sess = tf.Session()
sess.run(init)                                              # 运行

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))