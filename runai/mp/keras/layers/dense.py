import keras.backend as K
import keras.layers

from runai import log
import runai.mp

class Dense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        log.debug('Dense.__init__()')

    def build(self, input_shape):
        log.debug('Dense.build(input_shape=%s)', input_shape)

        assert len(input_shape) == 2 # TODO(levosos): support more than two dimensions
        
        cin = input_shape[-1]

        if runai.mp.method == runai.mp.Method.Cin:
            self.cin  = cin // runai.mp.splits # TODO(levosos): support uneven division
            self.cout = self.units
        elif runai.mp.method == runai.mp.Method.Cout:
            self.cin  = cin
            self.cout = self.units // runai.mp.splits # TODO(levosos): support uneven division

        log.info('Cin: %d -> %d; Cout: %d -> %d' % (cin, self.cin, self.units, self.cout))

        log.info('Declaring %d weights of shape (%d,%d) [instead of (%d,%d)]' % (runai.mp.splits, self.cin, self.cout, cin, self.units))
        self.kernels = [self.add_weight(
            shape=(self.cin, self.cout),
            initializer=self.kernel_initializer,
            name='kernel_%d' % i,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint # TODO(levosos): is this ok?
        ) for i in range(runai.mp.splits)]

        if self.use_bias:
            log.info('Declaring %d biases of shape (%d,) [instead of (%d,)]' % (runai.mp.splits, self.cout, self.units))
            self.biases = [self.add_weight(
                shape=(self.cout,),
                initializer=self.bias_initializer,
                name='bias_%d' % i,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint # TODO(levosos): is this ok?
            ) for i in range(runai.mp.splits)]
        
        self.input_spec = keras.layers.InputSpec(ndim=2, axes={-1: cin}) # TODO(levosos): use 'min_ndim' once supporting more than two dimensions
        self.built = True

    def call(self, inputs):
        log.debug('Dense.call(inputs=%s)', inputs)
        
        assert not isinstance(inputs, (tuple, list))

        def input(i):
            if runai.mp.method == runai.mp.Method.Cin:
                return inputs[:, self.cin * i : self.cin * (i + 1)]
            elif runai.mp.method == runai.mp.Method.Cout:
                return inputs
        
        def impl(i):
            output = K.dot(input(i), self.kernels[i])
            if self.use_bias:
                output = K.bias_add(output, self.biases[i], data_format='channels_last')
            if self.activation is not None:
                output = self.activation(output)
            return output

        duplications = [impl(i) for i in range(runai.mp.splits)]

        if runai.mp.method == runai.mp.Method.Cin:
            return keras.layers.Add()(duplications)
        elif runai.mp.method == runai.mp.Method.Cout:
            return K.concatenate(duplications, axis=-1)
        
    # TODO(levosos): implement get_config()
