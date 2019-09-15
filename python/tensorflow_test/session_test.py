"""
    session会话的两种形式
"""

import tensorflow as tf

matrix_1 = tf.constant([[3, 3]])
matrix_2 = tf.constant([[2], [2]])
print('matrix_1:', matrix_1)
print('matrix_2:', matrix_2)

product = tf.matmul(matrix_1, matrix_2)         # 矩阵乘法，类似np.dot(1. 2)


# method 1
sess = tf.Session()
result = sess.run(product)                      # 每run一次才会执行
print(result)
sess.close()


# method 2
with tf.Session() as sess:                      # 用with打开Session，最后自动关闭
    result_2 = sess.run(product)
    print(result_2)