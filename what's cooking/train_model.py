import tensorflow as tf
import numpy as np
import  time
import datetime
import math
import matplotlib.pyplot as plt

import create_df


class Train_Model(object):
    tf.set_random_seed(0)

    def __init__(self, filename, train_test_split=True):

        self.filename = filename
        self.train_test_split = train_test_split
        self.ids = []

        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None

        self.hidden_layer1 = dict()
        self.hidden_layer2 = dict()
        self.hidden_layer3 = dict()
        self.hidden_layer4 = dict()
        self.output_layer = dict()

        self.x = tf.placeholder(tf.float32, name='x')
        self.y = tf.placeholder(tf.float32, name='y')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.pkeep = tf.placeholder(tf.float32, name='pkeep')

    def get_data(self):
        if self.train_test_split:
            result = create_df.make_train_test_df(self.filename)
            self.train_features, self.train_labels, self.test_features, self.test_labels = result
        else:
            if self.filename == 'test.json':
                self.train_features, self.ids = create_df.make_df(self.filename)
            else:
                self.train_features, self.train_labels = create_df.make_df(self.filename)


    def layout(self):
        # self.train_features, self.train_labels, self.test_features, self.test_labels = self.get_data()
        self.get_data()

        self.hidden_layer1['nodes'] = 100
        self.hidden_layer1['weights'] = tf.Variable(
            tf.truncated_normal([len(self.train_features[0]), self.hidden_layer1['nodes']], stddev=0.1))
        self.hidden_layer1['bias'] = tf.Variable(tf.zeros([self.hidden_layer1['nodes']]))
        prev_nodes = self.hidden_layer1['nodes']

        self.hidden_layer2['nodes'] = 60
        self.hidden_layer2['weights'] = tf.Variable(
            tf.truncated_normal([prev_nodes, self.hidden_layer2['nodes']], stddev=0.1))
        self.hidden_layer2['bias'] = tf.Variable(tf.zeros([self.hidden_layer2['nodes']]))
        prev_nodes = self.hidden_layer2['nodes']

        self.hidden_layer3['nodes'] = 50
        self.hidden_layer3['weights'] = tf.Variable(
            tf.truncated_normal([prev_nodes, self.hidden_layer3['nodes']], stddev=0.1))
        self.hidden_layer3['bias'] = tf.Variable(tf.zeros([self.hidden_layer3['nodes']]))
        prev_nodes = self.hidden_layer3['nodes']

        self.hidden_layer4['nodes'] = 40
        self.hidden_layer4['weights'] = tf.Variable(
            tf.truncated_normal([prev_nodes, self.hidden_layer4['nodes']], stddev=0.1))
        self.hidden_layer4['bias'] = tf.Variable(tf.zeros([self.hidden_layer4['nodes']]))
        prev_nodes = self.hidden_layer4['nodes']

        self.output_layer['nodes'] = 20
        self.output_layer['weights'] = tf.Variable(
            tf.truncated_normal([prev_nodes, self.output_layer['nodes']], stddev=0.1))
        self.output_layer['bias'] = tf.Variable(tf.zeros([self.output_layer['nodes']]))

    def graph_gen(self, data):

        self.layout()

        layer1 = tf.add(tf.matmul(data, self.hidden_layer1['weights']), self.hidden_layer1['bias'])
        # layer1 = tf.nn.tanh(layer1)
        layer1 = tf.nn.relu(layer1)
        layer1 = tf.nn.dropout(layer1, self.pkeep)
        prev_layer = layer1

        layer2 = tf.add(tf.matmul(prev_layer, self.hidden_layer2['weights']), self.hidden_layer2['bias'])
        # layer2 = tf.nn.tanh(layer2)
        layer2 = tf.nn.relu(layer2)
        layer2 = tf.nn.dropout(layer2, self.pkeep)
        prev_layer = layer2

        layer3 = tf.add(tf.matmul(prev_layer, self.hidden_layer3['weights']), self.hidden_layer3['bias'])
        # layer3 = tf.nn.tanh(layer3)
        layer3 = tf.nn.relu(layer3)
        layer3 = tf.nn.dropout(layer3, self.pkeep)
        prev_layer = layer3

        layer4 = tf.add(tf.matmul(prev_layer, self.hidden_layer4['weights']), self.hidden_layer4['bias'])
        # layer4 = tf.nn.tanh(layer3)
        layer4 = tf.nn.relu(layer4)
        layer4 = tf.nn.dropout(layer4, self.pkeep)
        prev_layer = layer4

        output = tf.add(tf.matmul(prev_layer, self.output_layer['weights']), self.output_layer['bias'])

        return output

    def train_nn(self):

        # learning_rate = 0.0001
        beta = 0.01
        prediction = self.graph_gen(self.x)
        softmax_prediction = tf.nn.softmax(prediction)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        correct = tf.equal(tf.argmax(softmax_prediction, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        n_epochs = 200
        batch_size = 100

        train_acc_to_plot = []
        test_acc_to_plot = []
        cost_to_plot = []

        fig = plt.figure()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # plt.axis([0, n_epochs, 0, 1])
            # plt.ion()
            max_test_acc = 0.0

            for epoch in range(n_epochs):
                epoch_loss = 0
                i = 0

                # learning rate decay
                max_learning_rate = 0.01
                min_learning_rate = 0.0001
                decay_speed = 2000.0
                learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) \
                                                    * math.exp(-epoch / decay_speed)
                while i < len(self.train_features):
                    start = i
                    end = i + batch_size
                    batch_x = np.array(self.train_features[start:end])
                    batch_y = np.array(self.train_labels[start:end])

                    temp, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y,
                                                                     self.lr: learning_rate, self.pkeep: 0.85})

                    epoch_loss += c
                    i += batch_size

                cost_to_plot.append(epoch_loss)

                # if epoch % 5 == 0:
                train_accuracy = accuracy.eval(
                    {self.x: self.train_features, self.y: self.train_labels, self.pkeep: 1.0})
                train_acc_to_plot.append(train_accuracy)

                if self.train_test_split:
                    test_accuracy = accuracy.eval({self.x: self.test_features, self.y: self.test_labels,
                                                   self.pkeep: 1.0})
                    if test_accuracy > max_test_acc:
                        max_test_acc = test_accuracy
                    test_acc_to_plot.append(test_accuracy)

                    print 'epoch:', epoch, '\ttrain_accuracy:', train_accuracy, '\ttest_accuracy:', test_accuracy, \
                                        '\tloss:', epoch_loss, '\tmaximum test accuracy:', max_test_acc
                else:
                    print 'epoch:', epoch, '\ttrain_accuracy:', train_accuracy, \
                        '\tloss:', epoch_loss, '\tlearning_rate:', learning_rate


                    # plt.scatter(epoch + 1, epoch_loss, c='r')
            #     plt.scatter(epoch + 1, test_accuracy, c='y')
            #     plt.scatter(epoch + 1, train_accuracy, c='b')
            #     plt.pause(0.001)
            #
            # plt.ioff()
            # plt.show()
                if test_accuracy > 0.78:
                    break

            save_path = saver.save(sess, "./saved_models/model_v1.ckpt")
            print("Model saved in file: %s" % save_path)

        ax1 = fig.add_subplot(221)
        ax1.plot(range(n_epochs), train_acc_to_plot, label='train', c='b')
        ax1.set_ylim([0.97, 1.0])
        ax1.legend()

        if self.train_test_split:
            ax2 = fig.add_subplot(222)
            ax2.plot(range(n_epochs), test_acc_to_plot, label='test', c='y')
            ax2.set_ylim([0.75, 0.79])
            ax2.legend()

        ax3 = fig.add_subplot(223)
        ax3.plot(range(n_epochs), cost_to_plot, label='cost', c='r')
        ax3.set_ylim([0, 100])
        ax3.legend()

        plt.show()
        # print "maximum test accuracy:", max_test_acc

    def predict_nn(self, actual_data=None):

        prediction = self.graph_gen(self.x)
        print 'len:', len(self.train_features)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "./saved_models/model_v1.ckpt")
            output_file = './outputs/output' + str(datetime.datetime.now()) + '.csv'
            result = sess.run(tf.argmax(prediction.eval(feed_dict={self.x: self.train_features,
                                                                   self.pkeep: 1}), 1))
            with open(output_file, 'w') as f:
                f.write('id,cuisine\n')
                for i, item in enumerate(result):
                    if i % 1000 == 0:
                        print i, 'id:', self.ids[i], 'result:', result, create_df.one_hot_to_real(item)
                    f.write(str(self.ids[i]) + ',' + create_df.one_hot_to_real(item) + '\n')


train_model_obj = Train_Model('train.json')
train_model_obj.train_nn()

# train_model_obj = Train_Model('train.json', train_test_split=False)
# train_model_obj.train_nn()

# predict_model_obj = Train_Model('test.json', train_test_split=False)
# predict_model_obj.predict_nn()
