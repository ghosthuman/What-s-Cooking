import tensorflow as tf
import create_df
import numpy as np

class Train_Model(object):


    def __init__(self):
        self.train_features = None
        self.train_labels= None
        self.test_features= None
        self.test_labels= None
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

    def get_data(self):
        return create_df.make_df('train.json')

    def layout(self):
        self.train_features, self.train_labels, self.test_features, self.test_labels = self.get_data()

        print self.train_features[0].shape
        hidden_layer1 = dict()
        hidden_layer1['nodes'] = 6714
        hidden_layer1['weights'] = tf.Variable(tf.truncated_normal([len(self.train_features[0]),hidden_layer1['nodes']],stddev=0.1))
        hidden_layer1['bias'] = tf.Variable(tf.zeros([hidden_layer1['nodes']]))

        hidden_layer2 = dict()
        hidden_layer2['nodes'] = 3357
        hidden_layer2['weights'] = tf.Variable(tf.truncated_normal([hidden_layer1['nodes'],hidden_layer2['nodes']],stddev=0.1))
        hidden_layer2['bias'] = tf.Variable(tf.zeros([hidden_layer2['nodes']]))

        hidden_layer3 = dict()
        hidden_layer3['nodes'] = 1600
        hidden_layer3['weights'] = tf.Variable(tf.truncated_normal([hidden_layer2['nodes'],hidden_layer3['nodes']],stddev=0.1))
        hidden_layer3['bias'] = tf.Variable(tf.zeros([hidden_layer3['nodes']]))

        output_layer = dict()
        output_layer['nodes'] = 20
        output_layer['weights'] = tf.Variable(tf.truncated_normal([hidden_layer3['nodes'],output_layer['nodes']],stddev=0.1))
        output_layer['bias'] = tf.Variable(tf.zeros([output_layer['nodes']]))

        return hidden_layer1,hidden_layer2,hidden_layer3,output_layer

    def graph_gen(self, data):

        hidden_layer1, hidden_layer2 , hidden_layer3, output_layer = self.layout()
        layer1 = tf.add(tf.matmul(data,hidden_layer1['weights']), hidden_layer1['bias'])
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.add(tf.matmul(layer1, hidden_layer2['weights']), hidden_layer2['bias'])
        layer2 = tf.nn.relu(layer2)

        layer3 = tf.add(tf.matmul(layer2, hidden_layer3['weights']), hidden_layer3['bias'])
        layer3 = tf.nn.relu(layer3)

        output = tf.add(tf.matmul(layer3, output_layer['weights']), output_layer['bias'])

        return output

    def train_nn(self):

        learning_rate = 0.01
        prediction = self.graph_gen(self.x)
        softmax_prediction = tf.nn.softmax(prediction)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        correct = tf.equal(tf.argmax(softmax_prediction,1),tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

        n_epochs = 100
        batch_size = 100
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(n_epochs):
                epoch_loss = 0
                i = 0
                while i < len(self.train_features):
                    start = i
                    end = i + batch_size
                    batch_x = np.array(self.train_features[start:end])
                    batch_y = np.array(self.train_labels[start:end])

                    temp, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})

                    epoch_loss += c

                    i += batch_size

                train_accuracy = accuracy.eval({self.x:self.train_features, self.y:self.train_labels})

                print 'epoch:',epoch,'\taccuracy:',train_accuracy, '\tloss:', epoch_loss


train_model_obj = Train_Model()
train_model_obj.train_nn()








        


