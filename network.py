import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs=inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')

        self.eps = 1e-8
        self.kernels = {}
        self.biases = {}

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, 
                    initializer=tf.truncated_normal_initializer(stddev=5e-2), trainable=None):
        '''Creates a new TensorFlow variable.'''
        if trainable == None:
            trainable = self.trainable
        return tf.get_variable(name, shape, trainable=trainable, initializer=initializer)

    def make_var_with_weight_decay(self, name, shape, 
                    initializer=tf.truncated_normal_initializer(stddev=5e-2), trainable=None):
        if trainable == None:
            trainable = self.trainable
        var = tf.get_variable(name, shape, trainable=trainable, initializer=initializer)
        tf.add_to_collection('WEIGHT_DECAY_VARIABLE', var)
        return var

    def make_var_on_cpu(self, name, shape, 
                    initializer=tf.truncated_normal_initializer(stddev=5e-2), trainable=None):
        with tf.device('/cpu:0'):
            if trainable == None:
                trainable = self.trainable
            return tf.get_variable(name, shape, trainable=trainable, initializer=initializer)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert (padding in ('SAME', 'VALID')) or isinstance(padding, (int, long))

    @layer
    def bias(self,
             input,
             name):
        with tf.variable_scope(name) as scope:
            biases = self.make_var('biases', [input.get_shape()[-1]], initializer=tf.constant_initializer(0.))
            output = tf.nn.bias_add(input, biases)
        return output

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             biased=True):
        # Verify if the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var_with_weight_decay('weights', shape=[k_h, k_w, c_i, c_o])
            self.kernels[name] = kernel
            output = convolve(input, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o], initializer=tf.constant_initializer(0.))
                self.biases[name] = biases
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)

            tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output

    @layer
    def atrous_conv(self,
                    input, 
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        self.validate_padding(padding)

        c_i = input.get_shape()[-1]
        
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var_with_weight_decay('weights', shape=[k_h, k_w, c_i, c_o])
            self.kernels[name] = kernel
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o], initializer=tf.constant_initializer)
                self.biases[name] = biases
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output

    @layer
    def deconv(self, 
              input,
              conv_name,
              name,
              relu=True,
              padding=DEFAULT_PADDING,
              biased=True):

        self.validate_padding(padding)

        with tf.variable_scope(name) as scope:
            kernel = self.kernels[name]
            if biased:
                bias = self.biases[name]
                output = tf.nn.bias_add(input, bias)
            output = tf.nn.conv2d_transpose(output, kernel, output_shape, [1, s_h, s_w, 1], padding=padding)

            tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output

    @layer
    def deconv(self,
             input,
             k_h,
             k_w,
             s_h,
             s_w,
             output_shape,
             name,
             relu=True,
             padding=DEFAULT_PADDING):
        # Verify if the padding is acceptable
        self.validate_padding(padding)

        i_shape = input.get_shape().as_list()
        o_shape = output_shape.as_list()

        convolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape, [1, s_h, s_w, 1], padding=padding)

        # Convolution for a given input and kernel
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, o_shape[-1], i_shape[-1]])
            self.kernels[name] = kernel
            output = convolve(input, kernel)

            if relu:
                output = tf.nn.relu(output, name=scope.name)
            
            tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output 


    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def leaky_relu(self, input, name, alpha=0.2):
        return tf.nn.leaky_relu(input, alpha=alpha, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        output = tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)
        tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        output = tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)
        tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        output = tf.concat(values=inputs, axis=axis, name=name)
        tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def multiply(self, inputs, name):
        output = inputs[0]
        with tf.name_scope(name) as scope:
            for i in range(1, len(inputs)):
                output = tf.multiply(output, inputs[i], name + '_{}'.format(i))

        return output

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out], initializer=tf.constant_initializer(0.))
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)

            tf.logging.debug('in {}, output shape: {}'.format(name, fc.get_shape()))
            return fc

    @layer
    def operation(self, input, op, name):
        output = op(input, name)
        tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output

    @layer
    def softmax(self, input, name):
        return tf.nn.softmax(input, name=name)

    @layer
    def BN(self, input, slope, bias, name, eps=1e-8, momentum=0.95):
        from tensorflow.python.training import moving_averages
        with tf.variable_scope(name) as scope:

            shape = [input.get_shape()[-1]]

            beta = self.make_var('bias', shape=shape, 
                        initializer=tf.constant_initializer(bias))
            gamma = self.make_var('slope', shape=shape, 
                        initializer=tf.constant_initializer(slope))
            moving_mean = self.make_var('moving_mean', shape=shape, 
                        initializer=tf.zeros_initializer, trainable=False)
            moving_var = self.make_var('moving_var', shape=shape, 
                        initializer=tf.zeros_initializer, trainable=False)

            axes = [n for n in range(len(input.get_shape())-1)]
        
            mean, var = tf.nn.moments(input, axes=axes)

            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, momentum)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, momentum)

            tf.add_to_collection('UPDATE_OPS_COLLECTION', update_moving_mean)
            tf.add_to_collection('UPDATE_OPS_COLLECTION', update_moving_var)

            if not self.trainable:
                mean, var = moving_mean, moving_var


            output = tf.nn.batch_normalization(input, mean, var, beta, gamma, eps, name=name)

        return output

    @layer
    def batch_normalization(self, input, name, relu=False):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            output = tf.layers.batch_normalization(
                            input,
                            momentum=0.95,
                            epsilon=self.eps,
                            training=self.trainable,
                            name=name)
            if relu:
                output = tf.nn.relu(output)
        return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)


    @layer
    def reshape(self, input, shape, name):
        output = tf.reshape(input, shape)
        tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output

    @layer
    def interp(self, input, name, shape=None, shrink_factor=None, zoom_factor=None, height=None, width=None):
        o_shape = input.get_shape().as_list()
        o_h = o_shape[1]
        o_w = o_shape[2]

        if shape != None:
            o_h = shape[0]
            o_w = shape[1]
        elif shrink_factor != None:
            o_h = (o_h-1) / shrink_factor + 1
            o_w = (o_w-1) / shrink_factor + 1

        elif zoom_factor != None:
            o_h = o_h + (o_h-1) * (zoom_factor-1)
            o_w = o_w + (o_w-1) * (zoom_factor-1)

        elif height != None and width != None:
            o_h = height
            o_w = width

        output = tf.image.resize_images(input, [o_h, o_w], align_corners=True)
        tf.logging.debug('in {}, output shape: {}'.format(name, output.get_shape()))
        return output
