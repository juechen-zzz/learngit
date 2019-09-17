"""
    add layer function

    newaxis：    当把newaxis放在前面的时候，以前的shape是5，现在变成了1××5，也就是前面的维数发生了变化，后面的维数发生了变化

                而把newaxis放后面的时候，输出的新数组的shape就是5××1，也就是后面增加了一个维数
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


# make dataset
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # newaxis是为原数组增加一个维度，维度为300行，1列
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.compat.v1.placeholder(tf.float32, [None, 1])
ys = tf.compat.v1.placeholder(tf.float32, [None, 1])

# define layer
l_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
predict = add_layer(l_1, 10, 1, activation_function=None)

# reduction_indices=[1]是按行求和，reduce_sum求出的是一列值，reduce_mean是对这个一列值进行求均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predict), reduction_indices=[1]))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

# 建立图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()               # 在show后能够继续走下去


for i in range(1000):
    # 方便使用batch
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print('loss:', sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])       # 在图片中去除lines的第一个线段
        except Exception:
            pass
        predict_value = sess.run(predict, feed_dict={xs: x_data})
        lines = ax.plot(x_data, predict_value, 'r-', lw=5)
        plt.pause(0.1)                      # 暂停0.1秒

plt.ioff()
plt.show()
