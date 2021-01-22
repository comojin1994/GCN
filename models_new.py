import tensorflow as tf
from spektral.layers import GCNConv
import config as C

class GCN(tf.keras.Model):
    def __init__(self, filters, dropout):
        super(GCN, self).__init__(name='GCN')
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.gcn_1 = GCNConv(filters,
                            activation='relu',
                            use_bias=False,
                            kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        self.dropout_2 = tf.keras.layers.Dropout(dropout)
        self.gcn_2 = GCNConv(C.num_classes,
                            activation='softmax',
                            use_bias=False,
                            kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
    def call(self, input_tensor, training=False):
        x, A = input_tensor
        x = self.dropout_1(x)
        x = self.gcn_1([x, A])
        x = self.dropout_2(x)
        output = self.gcn_2([x, A])
        return output

class InceptionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, activation=None, isLast=False):
        super(InceptionBlock, self).__init__()
        
        self.gcn1 = GCNConv(filters,
                           activation=activation,
                           use_bias=False,
                           kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
        self.gcn2 = GCNConv(filters,
                           activation=activation,
                           use_bias=False,
                           kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
        self.gcn3 = GCNConv(filters,
                           activation=activation,
                           use_bias=False,
                           kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
        self.average = tf.keras.layers.Average()
        self.isLast = isLast
        
    def call(self, input_tensor, training=False):
        x, A = input_tensor
        A2 = tf.matmul(tf.sparse.to_dense(A, 0.0),
                      tf.sparse.to_dense(A, 0.0),
                      a_is_sparse=True, b_is_sparse=True)
        A3 = tf.matmul(A2,
                      tf.sparse.to_dense(A, 0.0),
                      a_is_sparse=False, b_is_sparse=True)
        f1 = self.gcn1([x, A])
        f2 = self.gcn2([x, A2])
        f3 = self.gcn3([x, A3])
        
        output = self.average([f1, f2, f3])
        
        if self.isLast:
            output = tf.nn.softmax(output)
        else:
            output = tf.nn.relu(output)
        
        return output

class InceptionGCN(tf.keras.Model):
    def __init__(self, dropout):
        super(InceptionGCN, self).__init__(name='Inception_GCN')
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.inception_1 = InceptionBlock(C.GCN_filters, activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(dropout)
        self.inception_2 = InceptionBlock(C.num_classes, activation='relu', isLast=True)
        
    def call(self, input_tensor, training=False):
        x, A = input_tensor
        x = self.dropout_1(x)
        x = self.inception_1([x, A])
        x = self.dropout_2(x)
        output = self.inception_2([x, A])
        return output

