import keras.backend as K
import keras.layers

import runai

from .keep import Keep
from .parallelised import Parallelised

class Dense(Parallelised, keras.layers.Dense):
    def build(self, input_shape):
        assert len(input_shape) == 2 # TODO(levosos): support more than two dimensions

        total_cin = input_shape[-1]

        cin, cout, c = self.calculate_cs(
            cin=total_cin,
            cout=self.units)

        self.kernels = self.add_weights(
            name='kernel',
            shape=(cin, cout),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint) # TODO(levosos): is this ok?

        if self.use_bias:
            self.biases = self.add_weights(
                name='bias',
                shape=(c,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint) # TODO(levosos): is this ok?
        
        self.input_spec = keras.layers.InputSpec(ndim=2, axes={-1: total_cin}) # TODO(levosos): use 'min_ndim' once supporting more than two dimensions
        self.built = True

    def call(self, inputs):
        inputs = self.inputs(inputs, channel_axis=-1)

        outputs = self.parallelise(
            lambda input, kernel: K.dot(input, kernel),
            inputs,
            self.kernels)

        if runai.mp.method == runai.mp.Method.Cin:
            outputs = self.reduce_split(outputs, channel_axis=-1)

        if self.use_bias:
            outputs = self.parallelise(
                lambda output, bias: K.bias_add(output, bias, data_format='channels_last'),
                outputs,
                self.biases)

        if self.activation is not None:
            outputs = self.parallelise(
                lambda output: self.activation(output),
                outputs)
        
        return self.merge(outputs, channel_axis=-1)

class Dropout(Keep, keras.layers.Dropout): pass
