import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
import config as C
from layers import GraphConv, InceptionBlock, ResidualBlock

class GCN(tf.keras.Model):
    def __init__(self, filters, dropout):
        super(GCN, self).__init__(name='GCN')
        self.dropout_1 = Dropout(dropout)
        self.graphConv_1 = GraphConv(filters,
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        self.dropout_2 = Dropout(dropout)
        self.graphConv_2 = GraphConv(C.num_classes,
                                    activation=tf.nn.softmax,
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
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
        self.dropout_1 = Dropout(dropout)
        self.dense_2 = Dense(filters[1],
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        self.dropout_2 = Dropout(dropout)
        self.dense_3 = Dense(C.num_classes,
                                            activation=tf.nn.softmax)

    def call(self, input_tensor, training=False):
        x = self.dense_1(input_tensor)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        output= self.dense_3(x)
        return output

class InceptionGCN(tf.keras.Model):
    def __init__(self, dropout):
        super(InceptionGCN, self).__init__(name='Inception_GCN')
        self.dropout_1 = Dropout(dropout)
        self.inception_1 = InceptionBlock(C.GCN_filters, activation=tf.nn.relu)
        self.dropout_2 = Dropout(dropout)
        self.inception_2 = InceptionBlock(C.num_classes, activation=tf.nn.relu, isLast=True)

    def call(self, input_tensor, training=False):
        A, x = input_tensor
        x = self.dropout_1(x)
        x = self.inception_1([A, x])
        x = self.dropout_2(x)
        x = self.inception_2([A, x])
        return x

class ResidualGCN(tf.keras.Model):
    def __init__(self, dropout):
        super(ResidualGCN, self).__init__(name='ResidualGCN')
        
        self.dropout_1 = Dropout(dropout)
        self.resBlock_1 = ResidualBlock(C.GCN_filters)
        self.dropout_2 = Dropout(dropout)
        self.resBlock_2 = ResidualBlock(C.num_classes)
        
    def call(self, input_tensor, training=False):
        A, x = input_tensor
        x = self.dropout_1(x)
        x = self.resBlock_1([A, x])
        x = tf.nn.relu(x)
        x = self.dropout_2(x)
        x = self.resBlock_2([A, x])
        output = tf.nn.softmax(x)
        return output

        