import tensorflow as tf
import config as C

class GraphConv(tf.keras.layers.Layer):
    def __init__(self, filters, activation=None, bn=False, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None):
        super(GraphConv, self).__init__()
        self.filters = filters
        self.activation = activation
        self.use_bn = bn
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.BatchNorm = tf.keras.layers.BatchNormalization()
        
    def build(self, input_shape):
        print(f'LOG >>> input shape: {input_shape}')
        self.weight = self.add_weight(name='weight',
                                       shape=[input_shape[1][-1], self.filters],
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                       shape=self.filters,
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       trainable=True)
        
    def call(self, inputs):
        # A = inputs[0]
        # x = inputs[1]
        x = tf.matmul(inputs[0], inputs[1])
        output = tf.matmul(x, self.weight)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.use_bn:
            output = self.BatchNorm(output)
        if self.activation:
            output = self.activation(output)
        return [inputs[0], output]

class Readout(tf.keras.layers.Layer):
    def __init__(self, filters, activation=None, mode='mlp'):
        super(Readout, self).__init__()
        
        self.filters = filters
        self.dense = tf.keras.layers.Dense(filters, use_bias=False)
        self.activation = activation
        self.mode = mode
        
    def call(self, x):
        print(f'LOG >>> x\n{x}')
        if self.mode == 'mean':
            x = tf.math.reduce_mean(x, axis=1)
            print(f'LOG >>> output\n{x}')
            output = tf.math.reduce_sum(x, axis=0)
        else:
            x = self.dense(x)
            print(f'LOG >>> weight\n{self.dense.weights}')
            print(f'LOG >>> output\n{x}')
            output = tf.math.reduce_sum(x, axis=1)
        
        if self.activation != None:
            output = self.activation(output)
        return output

class InceptionBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(InceptionBlock, self).__init__()

        self.gcn1 = GraphConv(filters,
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        
        self.gcn2_1 = GraphConv(filters,
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        self.gcn2_2 = GraphConv(filters,
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))

        self.gcn3_1 = GraphConv(filters,
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        self.gcn3_2 = GraphConv(filters,
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))
        self.gcn3_3 = GraphConv(filters,
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(C.l2_reg))

        self.average = tf.keras.layers.Average()

    def call(self, input_tensor, training=False):
        A, x = input_tensor
        A, f1 = self.gcn1([A, x])

        A, f2 = self.gcn2_1([A, x])
        A, f2 = self.gcn2_2([A, f2])

        A, f3 = self.gcn3_1([A, x])
        A, f3 = self.gcn3_2([A, f3])
        A, f3 = self.gcn3_3([A, f3])
        
        output = self.average([f1, f2, f3])        
        
        return output
        
        