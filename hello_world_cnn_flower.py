from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

path = 'C:\\Users\\fooww\\Desktop\\flower_photos'

# 图片大小
w = 100
h = 100
c = 3


# 读取图片
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path)]
    images = []
    labels = []

    for index, folder in enumerate(cate):
        for imagePath in glob.glob(folder + '/*.jpg'):
            print('reading the image:%s' % imagePath)
            image = io.imread(imagePath)
            image = transform.resize(image, (w, h))
            images.append(image)
            labels.append(index)
    return np.asanyarray(images, np.float32), np.asanyarray(labels, np.int32)


data, label = read_img(path)

# 打乱顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

# 将数据分为训练集和验证集
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
x_val = data[s:]
y_train = label[:s]
y_val = label[s:]

# 构建网络
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# 第一层卷积
# relu 小于0取0，大于0保留
# 常用激活函数
conv1 = tf.layers.conv2d(
    inputs=x, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
# 池化 50*50*32
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 第二层卷积
conv2 = tf.layers.conv2d(
    inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
# 池化 25*25*64
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 第三层卷积
conv3 = tf.layers.conv2d(
    inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
# 池化 12*12*128
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# 第四层卷积
conv4 = tf.layers.conv2d(
    inputs=pool3, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
# 池化 6*6*128
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

# 扁平化
rel = tf.reshape(pool4, [-1, 6 * 6 * 128])

# 全连接
dense1 = tf.layers.dense(inputs=rel, units=1024, activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu,
                         kernel_regularizer=tf.truncated_normal_initializer(stddev=0.01))

# 输出层
logits = tf.layers.dense(inputs=dense2, units=5, activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

# 损失函数
# argmax 返回每行或每列的最大索引，0 表示列 1 表示行
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

