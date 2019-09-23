"""
    dropout解决过拟合问题
"""

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 装载数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# 新增层函数
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None,):
    Weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # dropout
    Wx_plus_b = tf.compat.v1.nn.dropout(Wx_plus_b, rate=1-keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        tf.compat.v1.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

# 定义占位符
keep_prob = tf.compat.v1.placeholder(tf.float32)
xs = tf.compat.v1.placeholder(tf.float32, [None, 64])
ys = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 定义输出层
l_1 = add_layer(xs, 64, 50, 'L_1', activation_function=tf.nn.tanh)
prediction = add_layer(l_1, 50, 10, 'L_2', activation_function=tf.nn.softmax)

# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.math.log(prediction), reduction_indices=[1]))
tf.compat.v1.summary.scalar('loss', cross_entropy)
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

# 定义会话
sess = tf.compat.v1.Session()
merged = tf.compat.v1.summary.merge_all()

train_writer = tf.compat.v1.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.compat.v1.summary.FileWriter('logs/test', sess.graph)

sess.run(tf.compat.v1.global_variables_initializer())

for i in range(500):
    # here to determine the keeping probability
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})  # 50%被丢弃
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

# $ tensorboard --logdir=logs    先转到log之上的那个目录

