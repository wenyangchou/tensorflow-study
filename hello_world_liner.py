import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xdata = np.linspace(0, 1, 100)
ydata = 2 * xdata + 1

# 初始化变量
X = tf.placeholder("float", name="X")
Y = tf.placeholder("float", name="Y")
W = tf.Variable(3., name="W")
B = tf.Variable(3., name="B")

# 设定模型
linearmodel = tf.add(tf.multiply(W, X), B)

# 损失函数
lossfunc = (tf.pow(Y - linearmodel, 2))

# 训练
learningrate = 0.01
trainoperation = tf.train.GradientDescentOptimizer(learningrate).minimize(lossfunc)

# 初始化session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 开始训练
print("开始训练")
for j in range(100):
    for i in range(100):
        sess.run(trainoperation, feed_dict={X: xdata[i], Y: ydata[i]})

# 结果展示
plt.scatter(xdata, ydata)
plt.plot(xdata, B.eval(session=sess) + W.eval(session=sess) * xdata, 'b', label='calculated:w*x + b')
plt.legend()
plt.show()

print("训练完成")
print("权重w：" + str(W.eval(session=sess)))
print("偏移b:" + str(B.eval(session=sess)))
