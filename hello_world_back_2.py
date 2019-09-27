import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 创建计算图会话
sess = tf.Session()

# 生成数据并创建在占位符和变量A

x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

x_data = tf.placeholder(tf.float32, shape=[1])
y_target = tf.placeholder(tf.float32, shape=[1])

A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# 增加加法操作
my_output = tf.add(x_data, A)

# 由于非归一化logits的交叉熵的损失函数期望批量数据增加一个批量数据的维度
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

# 在运行之前，需要初始化变量
init = tf.initialize_all_variables()
sess.run(init)

# 增加非归一化logits的交叉熵的损失函数

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)

# 声明变量的优化器
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)

train_step = my_opt.minimize(loss)

num = 1500
step = np.zeros(num)
LOSS = np.zeros_like(step)
# 训练算法
for i in range(num):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    # 打印
    step[i] = i
    LOSS[i] = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 200 == 0:
        print('step =' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('loss =' + str(LOSS[i]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(step, LOSS, label='loss')
ax.set_xlabel('step')
ax.set_ylabel('loss')
fig.suptitle('sigmoid_cross_entropy_with_logits')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels=labels)
plt.show()

# logdir = './log'
# write = tf.summary.FileWriter(logdir=logdir,graph=sess.graph)