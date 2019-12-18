import keras.backend as K
import keras.layers

import runai.mp
import runai.utils

from .keep import Keep
from .parallelised import Parallelised

class Conv2D(Parallelised, keras.layers.Conv2D):
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1

        total_cin = input_shape[self.channel_axis]
        
        if total_cin < runai.mp.splits:
            runai.utils.log.warning('Not parallelising \'%s\' layer "%s" with input shape %s', self.__class__.__name__, getattr(self, 'name', 'N/A'), input_shape)
            self._parallelised = False
            return super(Conv2D, self).build(input_shape)

        self._parallelised = True

        cin, cout, c = self.calculate_cs(
            cin=total_cin,
            cout=self.filters)

        self.kernels = self.add_weights(
            name='kernel',
            shape=self.kernel_size + (cin, cout),
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

        self.input_spec = keras.layers.InputSpec(ndim=self.rank + 2, axes={self.channel_axis: total_cin})
        self.built = True

    def call(self, inputs):
        if not self._parallelised:
            return super(Conv2D, self).call(inputs)
        
        inputs = self.inputs(inputs, channel_axis=self.channel_axis)

        assert self.rank == 2 # TODO(levosos): support other convolutions

        outputs = self.parallelise(
            lambda input, kernel: K.conv2d(
                input,
                kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate),
            inputs,
            self.kernels)
        
        if runai.mp.method == runai.mp.Method.Cin:
            outputs = self.reduce_split(outputs, channel_axis=self.channel_axis)

        if self.use_bias:
            outputs = self.parallelise(
                lambda output, bias: K.bias_add(output, bias, data_format=self.data_format),
                outputs,
                self.biases)

        if self.activation is not None:
            outputs = self.parallelise(
                lambda output: self.activation(output),
                outputs)
        
        return self.merge(outputs, channel_axis=self.channel_axis)

ZeroPadding2D = Keep.create('ZeroPadding2D')
