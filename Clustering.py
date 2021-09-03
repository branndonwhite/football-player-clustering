from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

class Clustering:
    def __init__(self, height, width, input_dim):
        self.height = height
        self.width = width
        self.input_dim = input_dim
        self.weight = tf.Variable(tf.random_normal([height * width, input_dim]))
        self.input = tf.placeholder(tf.float32, [input_dim])
        self.location = tf.cast([[y, x] for y in range(height) for x in range(width)], tf.float32)
        self.winner = self.get_winner()
        self.updated_weight = self.update_weight()

    def get_winner(self):
        square_diff = tf.square(self.input - self.weight)
        euclidean = tf.sqrt(tf.reduce_sum(square_diff, axis=1))
        winning_unit = tf.argmin(euclidean)
        winning_loc = tf.cast([tf.div(winning_unit, self.height), tf.mod(winning_unit, self.width)], tf.float32)

        return winning_loc

    def update_weight(self):
        square_diff = tf.square(self.winner - self.location)
        euclidean = tf.sqrt(tf.reduce_sum(square_diff, axis=1))
        sigma = tf.cast(tf.maximum(self.height, self.width) / 2, tf.float32)
        ns = tf.exp((tf.negative(tf.square(euclidean)))/(2 * tf.square(sigma)))

        lr = .2
        rate = ns * lr
        rate_stacked = tf.stack([tf.tile([rate[i]], [self.input_dim]) for i in range(self.width * self.height)])

        new_weight = self.weight + (rate_stacked * (self.input - self.weight))

        return tf.assign(self.weight, new_weight)

    def train(self, dataset, epoch=5000):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                for data in dataset:
                    sess.run(self.updated_weight, feed_dict={self.input : data})
            
            weight = sess.run(self.weight)
            location = sess.run(self.location)

            cluster = [[] for i in range(self.height)]

            for i, loc in enumerate(location):
                cluster[int(loc[0])].append(weight[i])

            self.cluster = cluster

data = pd.read_csv("O202-COMP7117-VJ03-00-clustering.csv")

feature = data[['Preferred Foot', 'Jersey Number', 'Strength', 'Aggression', 'Interceptions', 'Positioning', 'Composure']]

feature = feature.dropna()

ordinal_enc = OrdinalEncoder()
feature[['Preferred Foot']] = ordinal_enc.fit_transform(feature[['Preferred Foot']])

feature = feature.to_numpy().astype(float)
feature = np.array(feature)

scaler = MinMaxScaler()
feature = scaler.fit_transform(feature)

mean = tf.reduce_mean(feature, axis=0)
centered_dataset = feature - mean

with tf.Session() as sess:
    dataset = sess.run(centered_dataset)

    pca = PCA(n_components=4)
    new_dataset = pca.fit_transform(dataset)

height = 4
width = 4
dimension = 4

som = Clustering(height, width, dimension)
som.train(new_dataset)

plt.imshow(som.cluster)
plt.show()