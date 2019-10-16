import keras.backend as K
import keras.layers
import tensorflow as tf

from runai import log
import runai.mp

from . import coordinator

class Conv2D(keras.layers.Conv2D):
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1

        cin  = input_shape[self.channel_axis]
        cout = self.filters

        if runai.mp.method == runai.mp.Method.Cin:
            self.cin  = cin // runai.mp.splits # TODO(levosos): support uneven division
            self.cout = cout
        elif runai.mp.method == runai.mp.Method.Cout:
            self.cin  = cin
            self.cout = cout // runai.mp.splits # TODO(levosos): support uneven division

        kernel_shape = self.kernel_size + (self.cin, self.cout)

        log.debug('Declaring %d weights of shape %s for conv layer \'%s\'' % (runai.mp.splits, kernel_shape, self.name))

        self.kernels = [self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel_%d' % i,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint # TODO(levosos): is this ok?
        ) for i in range(runai.mp.splits)]
        
        if self.use_bias:
            size = self.cout
            
            if runai.mp.method == runai.mp.Method.Cin:
                # biases are added after the reduce-split
                # therefore, in both cin and cout it will be splitted
                size = size // runai.mp.splits
            
            log.debug('Declaring %d biases of shape (%d,) for conv layer \'%s\'' % (runai.mp.splits, size, self.name))

            self.biases = [self.add_weight(
                shape=(size,),
                initializer=self.bias_initializer,
                name='bias_%d' % i,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint # TODO(levosos): is this ok?
            ) for i in range(runai.mp.splits)]

        self.input_spec = keras.layers.InputSpec(ndim=self.rank + 2, axes={self.channel_axis: cin})
        self.built = True

    def call(self, inputs):
        assert not isinstance(inputs, (tuple, list))
        
        if runai.mp.method == runai.mp.Method.Cin:
            if coordinator.registered(inputs):
                log.info('Using parallelised input for conv layer \'%s\'', self.name)
                inputs = coordinator.resolve(inputs)
            else:
                log.info('Splitting non-parallelised input for conv layer \'%s\'', self.name)
                inputs = tf.split(inputs, runai.mp.splits, axis=self.channel_axis)
        elif runai.mp.method == runai.mp.Method.Cout:
            inputs = [inputs] * runai.mp.splits

        assert self.rank == 2 # TODO(levosos): support other convolutions

        outputs = [K.conv2d(
            input,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        ) for input, kernel in zip(inputs, self.kernels)] # TODO(levosos): device placement

        if runai.mp.method == runai.mp.Method.Cin:
            reduced = keras.layers.Add()(outputs) # TODO(levosos): implement better reduce-split
            outputs = tf.split(reduced, runai.mp.splits, axis=self.channel_axis)

        if self.use_bias:
            outputs = [K.bias_add(output, bias, data_format=self.data_format) for output, bias in zip(outputs, self.biases)] # TODO(levosos): device placement

        if self.activation is not None:
            outputs = [self.activation(output) for output in outputs] # TODO(levosos): device placement
        
        merged = keras.layers.Concatenate(axis=self.channel_axis)(outputs)
        coordinator.register(merged, outputs)
        return merged
