import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
import config as C
from layers import GraphConv

class GCN(tf.keras.Model):
    def __init__(self, filters, dropout):
        super(GCN, self).__init__(name='GCN')
        self.dropout_1 = Dropout(C.dropout)
        self.graphConv_1 = GraphConv(filters,
                                    activation=tf.nn.relu,
                                    use_bias=False)
        self.dropout_2 = Dropout(C.dropout)
        self.graphConv_2 = GraphConv(C.num_classes,
                                    activation=tf.nn.softmax,
                                    use_bias=False)
        
    def call(self, input_tensor, training=False):
        A, x = input_tensor
        x = self.dropout_1(x)
        A, x = self.graphConv_1([A, x])
        x = self.dropout_2(x)
        A, x = self.graphConv_2([A, x])
        return x
 
class DNN(tf.keras.Model):
    def __init__(self, filters, dropout):
        super(DNN, self).__init__(name='DNN')
        self.dense_1 = Dense(filters[0],
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        self.dropout_1 = Dropout(C.dropout)
        self.dense_2 = Dense(filters[1],
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        self.dropout_2 = Dropout(C.dropout)
        self.dense_3 = Dense(C.num_classes,
                                            activation=tf.nn.softmax)

    def call(self, input_tensor, training=False):
        x = self.dense_1(input_tensor)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        output= self.dense_3(x)
        return output