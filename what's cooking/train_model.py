import tensorflow as tf
import create_df

class train_model(object):

    def get_data(self):
        return create_df.make_df('train.json')

    def layout(self):
        hidden_layer1 = dict()
        hidden_layer1['nodes'] = 6714
        hidden_layer1['weights'] =
        hidden_layer1['bias'] =

        hidden_layer2 = dict()
        hidden_layer2['nodes'] = 3357
        hidden_layer2['weights'] =
        hidden_layer2['bias'] =

        hidden_layer3 = dict()
        hidden_layer3['nodes'] = 1600
        hidden_layer3['weights'] =
        hidden_layer3['bias'] =


    def graph_gen(self):
