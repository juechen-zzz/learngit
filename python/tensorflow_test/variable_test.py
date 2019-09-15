"""
    Variable变量，一定要定义为变量才会是变量
"""

import tensorflow as tf

state = tf.Variable(0, name='counter')
print(state)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)                # 将new_value值加载到state中

init = tf.initialize_all_variables()                # 初始化所有变量，必须!

with tf.Session() as sess:
    sess.run(init)                                  # 执行初始化
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))                      # 直接print没用，必须把state指针放到run中才有用

