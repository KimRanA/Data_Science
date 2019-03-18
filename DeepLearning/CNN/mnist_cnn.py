# MNIST(Kaggle) with CNN ( class를 이용한 ensemble )
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# reset tensorflow graph
tf.reset_default_graph()


# Class Definition
class CnnModel:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.build_net()

    def build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
            self.Y = tf.placeholder(shape=[None, 10], dtype=tf.float32)
            self.keep_prob = tf.placeholder(dtype=tf.float32)

            ## Convolution layer
            X_img = tf.reshape(self.X, shape=[-1, 28, 28, 1])

            #             #     Filter를 생성
            #             W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
            #             #     Convolution
            #             L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding="SAME")
            #             #     ReLU
            #             L1 = tf.nn.relu(L1)
            #             #     MAX Pooling
            #             L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1],
            #                                 strides=[1,2,2,1], padding="SAME")

            L1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                  padding="SAME", strides=1, activation=tf.nn.relu)
            L1 = tf.layers.max_pooling2d(inputs=L1, pool_size=[2, 2],
                                         padding="SAME", strides=2)
            L1 = tf.layers.dropout(inputs=L1, rate=0.3)

            L2 = tf.layers.conv2d(inputs=L1, filters=64, kernel_size=[3, 3],
                                  padding="SAME", strides=1, activation=tf.nn.relu)
            L2 = tf.layers.max_pooling2d(inputs=L2, pool_size=[2, 2],
                                         padding="SAME", strides=2)
            L2 = tf.layers.dropout(inputs=L2, rate=0.3)

            L3 = tf.layers.conv2d(inputs=L2, filters=128, kernel_size=[3, 3],
                                  padding="SAME", strides=1, activation=tf.nn.relu)
            L3 = tf.layers.max_pooling2d(inputs=L3, pool_size=[2, 2],
                                         padding="SAME", strides=2)
            L3 = tf.layers.dropout(inputs=L3, rate=0.3)

            L3 = tf.reshape(L3, shape=[-1, 4 * 4 * 128])

            #             W1 = tf.get_variable("weight1", shape=[4*4*128,256],
            #                                 initializer=tf.contrib.layers.xavier_initializer())
            #             b1 = tf.Variable(tf.random_normal([256]), name="bias1")
            #             _layer1 = tf.nn.relu(tf.matmul(L2,W1) + b1)
            #             layer1 = tf.layers.dropout(_layer1, rate=self.keep_prob)

            ## dense layer
            dense1 = tf.layers.dense(inputs=L3,
                                     units=128,
                                     activation=tf.nn.relu)
            dense1 = tf.layers.dropout(inputs=dense1,
                                       rate=self.keep_prob)

            dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
            dense2 = tf.layers.dropout(inputs=dense2, rate=self.keep_prob)

            dense3 = tf.layers.dense(inputs=dense2, units=128, activation=tf.nn.relu)
            dense3 = tf.layers.dropout(inputs=dense3, rate=self.keep_prob)

            dense4 = tf.layers.dense(inputs=dense3, units=512, activation=tf.nn.relu)
            dense4 = tf.layers.dropout(inputs=dense4, rate=self.keep_prob)

            dense5 = tf.layers.dense(inputs=dense4, units=1024, activation=tf.nn.relu)
            dense5 = tf.layers.dropout(inputs=dense5, rate=self.keep_prob)

            #             self.H = tf.matmul(layer2, W3) + b3
            self.H = tf.layers.dense(inputs=dense5, units=10)
            # FC Layer ( Neural Network )
        #             W1 = tf.get_variable("weight1", shape=[7*7*64,256],
        #                                 initializer=tf.contrib.layers.xavier_initializer())
        #             b1 = tf.Variable(tf.random_normal([256]), name="bias1")
        #             _layer1 = tf.nn.relu(tf.matmul(L2,W1) + b1)
        #             layer1 = tf.layers.dropout(_layer1, rate=self.keep_prob)

        #             W2 = tf.get_variable("weight2", shape=[256,256],
        #                                 initializer=tf.contrib.layers.xavier_initializer())
        #             b2 = tf.Variable(tf.random_normal([256]), name="bias2")
        #             _layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
        #             layer2 = tf.layers.dropout(_layer2, rate=self.keep_prob)

        #             W3 = tf.get_variable("weight3", shape=[256,10],
        #                                 initializer=tf.contrib.layers.xavier_initializer())
        #             b3 = tf.Variable(tf.random_normal([10]), name="bias3")

        #             self.H = tf.matmul(layer2, W3) + b3

        #         self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.H, labels=self.Y))
        #         self.train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

        self.cost = tf.losses.softmax_cross_entropy(self.Y,
                                                    self.H)
        self.train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

        self.predict = tf.argmax(self.H, 1)
        self.correct = tf.equal(self.predict, tf.argmax(self.Y, 1))
        self.correct_count = tf.reduce_sum(tf.cast(self.correct, dtype=tf.float32))

    def train_net(self, train_x, train_y, rate):
        _, cost_val = self.sess.run([self.train, self.cost], feed_dict={self.X: train_x,
                                                                        self.Y: train_y,
                                                                        self.keep_prob: rate})

    def get_prediction(self, test_x, rate):
        H_val = self.sess.run(self.H, feed_dict={self.X: test_x,
                                                 self.keep_prob: rate})
        return H_val

    def get_accuracy(self, test_x, test_y, rate):
        return self.sess.run(self.correct_count, feed_dict={self.X: test_x,
                                                            self.Y: test_y,
                                                            self.keep_prob: rate})


## 1. Data Loading
############# MNIST DATA
# mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

# train_x_data = mnist.train.images
# test_x_data = mnist.test.images

# sess = tf.Session()

# train_y_data = mnist.train.labels
# test_y_data = mnist.test.labels

############# Kaggle Data ( accuracy 측정을 위한 7:3 분할 )
# data = pd.read_csv("./data/kaggle_mnist/train.csv", sep=",")

# data_x = data.drop("label", axis=1, inplace=False)
# data_y = data["label"]

# sess = tf.Session()
# num_of_train = int(data.shape[0] * 0.7) # 70%

# train_x_data = data_x.loc[:num_of_train,:].values
# test_x_data = data_x.loc[num_of_train+1:,:].values

# train_y_data = sess.run(tf.one_hot(data_y[:num_of_train+1].values,10))
# test_y_data = sess.run(tf.one_hot(data_y[num_of_train+1:].values, 10))

############# Kaggle Data ( train data를 모두 이용해서 학습 )
data = pd.read_csv("./data/kaggle_mnist/train.csv", sep=",")

data_x = data.drop("label", axis=1, inplace=False)
data_y = data["label"]

sess = tf.Session()
num_of_train = int(data.shape[0] * 0.7)  # 70%

train_x_data = data_x.loc[:, :].values
test_x_data = data_x.loc[num_of_train + 1:, :].values

train_y_data = sess.run(tf.one_hot(data_y[:].values, 10))
test_y_data = sess.run(tf.one_hot(data_y[num_of_train + 1:].values, 10))

## 3. Model 객체 생성
num_of_models = 20

models = [CnnModel(sess, "model" + str(x)) for x in range(num_of_models)]

## 4. 초기화
sess.run(tf.global_variables_initializer())

## 5. Model 학습
num_of_epoch = 20
batch_size = 100
keep_rate = 0.5

for model_idx in range(num_of_models):
    for step in range(num_of_epoch):
        num_of_iter = int(train_x_data.shape[0] / batch_size)
        idx = 0

        for i in range(num_of_iter):
            batch_x = train_x_data[idx:idx + batch_size, :]
            batch_y = train_y_data[idx:idx + batch_size, :]
            idx = idx + batch_size
            models[model_idx].train_net(batch_x, batch_y, keep_rate)

    print("Model {} 학습완료!!".format(model_idx))

## 6. 각 model의 Accuracy 측정
keep_rate = 1
num_of_iter = int(test_x_data.shape[0] / batch_size)

for model_idx in range(num_of_models):
    correct_count = 0
    idx = 0
    for i in range(num_of_iter):
        batch_x = test_x_data[idx:idx + batch_size, :]
        batch_y = test_y_data[idx:idx + batch_size, :]
        idx = idx + batch_size
        count = models[model_idx].get_accuracy(batch_x, batch_y, keep_rate)
        correct_count += count
    print("Model{} - Accuracy : {}".format(model_idx, correct_count / test_x_data.shape[0]))

## 7. ensemble의 Accuracy 측정
keep_rate = 1
idx = 0
correct_sum = 0
num_of_iter = int(test_x_data.shape[0] / batch_size)

l_correct_count = 0

for i in range(num_of_iter):
    batch_x = test_x_data[idx:idx + batch_size, :]
    batch_y = test_y_data[idx:idx + batch_size, :]
    idx = idx + batch_size

    prediction = np.zeros([batch_size, 10])

    for model_idx in range(num_of_models):
        p = models[model_idx].get_prediction(batch_x, keep_rate)
        prediction += p
    ## prediction을 구했어요

    l_predict = tf.argmax(prediction, 1)
    l_correct = tf.equal(l_predict, tf.argmax(batch_y, 1))
    l_count = tf.reduce_sum(tf.cast(l_correct, dtype=tf.float32))

    l_count_result = sess.run(l_count)
    l_correct_count += l_count_result

print("Accuracy : {}".format(l_correct_count / test_x_data.shape[0]))

## 8. 결과 도출

data = pd.read_csv("./data/kaggle_mnist/test.csv", sep=",")

test_x_data = data.values

keep_rate = 1
idx = 0
correct_sum = 0
num_of_iter = int(test_x_data.shape[0] / batch_size)

l_correct_count = 0
result = []

for i in range(num_of_iter):
    batch_x = test_x_data[idx:idx + batch_size, :]
    batch_y = test_y_data[idx:idx + batch_size, :]
    idx = idx + batch_size

    prediction = np.zeros([batch_size, 10])

    for model_idx in range(num_of_models):
        p = models[model_idx].get_prediction(batch_x, keep_rate)
        prediction += p
    #     ## prediction을 구했어요

    l_predict = tf.argmax(prediction, 1)
    tmp = sess.run(l_predict)
    result.extend(tmp)


df1 = pd.DataFrame([x + 1 for x in range(test_x_data.shape[0])], columns=["ImageId"])
df2 = pd.DataFrame(result, columns=["Label"])
df3 = df1.join(df2)
df3.to_csv("./data/kaggle_mnist/submission.csv", index=False)