import tensorflow as tf
import numpy as np

sess = tf.Session()

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

x_data = tf.placeholder(tf.float32, shape=[1])
y_target = tf.placeholder(tf.float32, shape=[1])

A = tf.Variable(tf.random_normal(shape=[1]))

my_output = tf.multiply(x_data, A)

# 损失函数
loss = tf.square(my_output - y_target)

# 初始化变量
init = tf.initialize_all_variables()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

# 训练算法
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i + 1) % 25 == 0:
        print("step = " + str(i + 1) + "A = " + str(sess.run(A)))
        print("loss = " + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

print(str(A.eval(session=sess)))
