"""
    将tensorflow代码进行可视化，看到模型结构，进行可视化的训练
"""

import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random.normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


# 定义名字的象征inputs
with tf.name_scope('inputs'):
    xs = tf.compat.v1.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l_1, 10, 1, activation_function=None)

# the error
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.compat.v1.Session()

# 最重要的一步，写出总结，加载到一个文件中
# $ tensorboard --logdir log    先转到log之上的那个目录
writer = tf.compat.v1.summary.FileWriter('./log', sess.graph)

sess.run(tf.compat.v1.global_variables_initializer())
