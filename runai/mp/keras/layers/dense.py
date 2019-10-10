import keras.backend as K
import keras.layers
import tensorflow as tf

from runai import log
import runai.mp

from . import coordinator

class Dense(keras.layers.Dense):
    def build(self, input_shape):
        assert len(input_shape) == 2 # TODO(levosos): support more than two dimensions
        
        cin  = input_shape[-1]
        cout = self.units

        if runai.mp.method == runai.mp.Method.Cin:
            self.cin  = cin // runai.mp.splits # TODO(levosos): support uneven division
            self.cout = cout
        elif runai.mp.method == runai.mp.Method.Cout:
            self.cin  = cin
            self.cout = cout // runai.mp.splits # TODO(levosos): support uneven division

        log.debug('Declaring %d weights of shape (%d,%d) for dense layer \'%s\'' % (runai.mp.splits, self.cin, self.cout, self.name))

        self.kernels = [self.add_weight(
            shape=(self.cin, self.cout),
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
            
            log.debug('Declaring %d biases of shape (%d,) for dense layer \'%s\'' % (runai.mp.splits, size, self.name))
            
            self.biases = [self.add_weight(
                shape=(size,),
                initializer=self.bias_initializer,
                name='bias_%d' % i,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint # TODO(levosos): is this ok?
            ) for i in range(runai.mp.splits)]
        
        self.input_spec = keras.layers.InputSpec(ndim=2, axes={-1: cin}) # TODO(levosos): use 'min_ndim' once supporting more than two dimensions
        self.built = True

    def call(self, inputs):
        assert not isinstance(inputs, (tuple, list))
        
        if runai.mp.method == runai.mp.Method.Cin:
            if coordinator.registered(inputs):
                log.info('Using parallelised input for dense layer \'%s\'', self.name)
                inputs = coordinator.resolve(inputs)
            else:
                log.info('Splitting non-parallelised input for layer \'%s\'', self.name)
                inputs = tf.split(inputs, runai.mp.splits, axis=-1)
        elif runai.mp.method == runai.mp.Method.Cout:
            inputs = [inputs] * runai.mp.splits

        outputs = [K.dot(input, kernel) for input, kernel in zip(inputs, self.kernels)] # TODO(levosos): device placement

        if runai.mp.method == runai.mp.Method.Cin:
            reduced = keras.layers.Add()(outputs) # TODO(levosos): implement better reduce-split
            outputs = tf.split(reduced, runai.mp.splits, axis=-1)

        if self.use_bias:
            outputs = [K.bias_add(output, bias, data_format='channels_last') for output, bias in zip(outputs, self.biases)] # TODO(levosos): device placement

        if self.activation is not None:
            outputs = [self.activation(output) for output in outputs] # TODO(levosos): device placement
        
        merged = keras.layers.Concatenate(axis=-1)(outputs)
        coordinator.register(merged, outputs)
        return merged
