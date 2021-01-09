import tensorflow as tf

class GraphConv(tf.keras.layers.Layer):
    def __init__(self, filters, activation=None, bn=False, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(GraphConv, self).__init__()
        self.filters = filters
        self.activation = activation
        self.use_bn = bn
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.BatchNorm = tf.keras.layers.BatchNormalization()
        
    def build(self, input_shape):
        self.weight = self.add_weight(name='weight',
                                       shape=[input_shape[1][-1], self.filters],
                                       initializer=self.kernel_initializer,
                                       trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                       shape=self.filters,
                                       initializer=self.bias_initializer,
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