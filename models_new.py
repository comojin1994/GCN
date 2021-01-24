import tensorflow as tf
from spektral.layers import GCNConv, GATConv
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Dropout
import config as C

# class GCN(tf.keras.Model):
#     def __init__(self, filters, dropout):
#         super(GCN, self).__init__(name='GCN')
#         self.dropout_1 = tf.keras.layers.Dropout(dropout)
#         self.gcn_1 = GCNConv(filters,
#                             activation='relu',
#                             use_bias=False,
#                             kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
#         self.dropout_2 = tf.keras.layers.Dropout(dropout)
#         self.gcn_2 = GCNConv(C.num_classes,
#                             activation='softmax',
#                             use_bias=False,
#                             kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
#     def call(self, input_tensor, training=False):
#         x, A = input_tensor
#         x = self.dropout_1(x)
#         x = self.gcn_1([x, A])
#         x = self.dropout_2(x)
#         output = self.gcn_2([x, A])
#         return output

class InceptionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, conv_activation=None):
        super(InceptionBlock, self).__init__()
        self.conv_activation = activations.get(conv_activation)

        self.gcn1 = GCNConv(filters,
                           activation=self.conv_activation,
                           use_bias=False,
                           kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
        self.gcn2 = GCNConv(filters,
                           activation=self.conv_activation,
                           use_bias=False,
                           kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
        self.gcn3 = GCNConv(filters,
                           activation=self.conv_activation,
                           use_bias=False,
                           kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
        self.average = tf.keras.layers.Average()
        
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
        
        return output

class InceptionGCN(tf.keras.Model):
    def __init__(self, dropout):
        super(InceptionGCN, self).__init__(name='Inception_GCN')
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.inception_1 = InceptionBlock(C.GCN_filters, activation='relu', conv_activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(dropout)
        self.inception_2 = InceptionBlock(C.num_classes, activation='softmax', conv_activation='relu')
        
    def call(self, input_tensor, training=False):
        x, A = input_tensor
        x = self.dropout_1(x)
        x = self.inception_1([x, A])
        x = self.dropout_2(x)
        output = self.inception_2([x, A])
        return output

class GraphConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                filters,
                activation=None,
                conv_activation=None,
                mode=None,
                sc=None,
                attn_heads=1,
                dropout=0.5,
                concat_heads=False,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                attn_kernel_regularizer=None,
                bias_regularizer=None):
        ### mode: 'inception', 'attention', None
        ### sc: 'sc', 'gsc', None
        ### concat_head: True, False
        super(GraphConvBlock, self).__init__()
        
        self.filters = filters
        self.activation = activations.get(activation)
        self.conv_activation = activations.get(conv_activation)
        self.mode = mode
        self.sc = sc
        
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        
        if self.mode == 'inception':
            self.layer = InceptionBlock(filters,
                                       conv_activation=self.conv_activation)
        elif self.mode == 'attention':
            self.layer = GATConv(filters,
                                attn_heads=self.attn_heads,
                                concat_heads=self.concat_heads,
                                dropout_rate=dropout,
                                activation=self.conv_activation,
                                kernel_regularizer=self.kernel_regularizer,
                                attn_kernel_regularizer=self.attn_kernel_regularizer,
                                bias_regularizer=self.bias_regularizer
                                )
        else:
            self.layer = GCNConv(filters,
                                activation=self.conv_activation,
                                use_bias=False,
                                kernel_regularizer=self.kernel_regularizer)
        
        self.dense = tf.keras.layers.Dense(filters, use_bias=False)        
        
    def build(self, input_shape):
        print(f'LOG >> input shape: {input_shape}')
        self.weight_in = self.add_weight(name='weight_in',
                                       shape=(input_shape[0][0], input_shape[0][0]),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True)
        self.bias_in = self.add_weight(name='bias_in',
                                       shape=(self.filters,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       trainable=True)
        self.weight_out = self.add_weight(name='weight_out',
                                       shape=(input_shape[0][0], input_shape[0][0]),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True)
        self.bias_out = self.add_weight(name='bias_out',
                                       shape=(self.filters,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       trainable=True)
        
    def call(self, input_tensor, training=False):
        x, A = input_tensor
        
        f = self.layer([x, A])
        
        if tf.shape(x)[-1] != tf.shape(f)[-1]:
            x = self.dense(x)
            
        if self.sc == 'gsc':            
            z = self.get_coefficient(x, f)
            output = tf.math.multiply(z, f) + tf.math.multiply(1 - z, x)
        elif self.sc == 'sc':
            output = x + f
        else:
            output = f
        
        output = self.activation(output)
        
        return output
            
    def get_coefficient(self, x, f):
        x = tf.matmul(self.weight_in, x) + self.bias_in
        f = tf.matmul(self.weight_out, f) + self.bias_out
        output = tf.nn.sigmoid(x + f)
        return output 

class GCN(tf.keras.Model):
    def __init__(self,
                dropout,
                activation=None,
                conv_activation=None,
                mode=None,
                sc=None,
                attn_heads=1,
                concat_heads=False,
                ):
        super(GCN, self).__init__(name=f'{mode}_{sc}_GCN')
        
        self.dropout_1 = Dropout(dropout)
        self.gcnBlock_1 = GraphConvBlock(C.GCN_filters,
                                        activation=activation,
                                        conv_activation=conv_activation,
                                        mode=mode,
                                        sc=sc,
                                        attn_heads=attn_heads,
                                        concat_heads=concat_heads)

        self.dropout_2 = Dropout(dropout)
        self.gcnBlock_2 = GraphConvBlock(C.num_classes,
                                        activation='softmax',
                                        conv_activation=conv_activation,
                                        mode=mode,
                                        sc=sc,
                                        attn_heads=attn_heads,
                                        concat_heads=concat_heads)
        
    def call(self, input_tensor, training=False):
        x, A = input_tensor
        x = self.dropout_1(x)
        x = self.gcnBlock_1([x, A])
        x = self.dropout_2(x)
        output = self.gcnBlock_2([x, A])
        return output