import keras.layers
import tensorflow as tf

from runai import log

from . import coordinator

class Keep(keras.layers.Layer):
    """ A MP-supported generic Keras layer

    The main goal of this class is to ease the implementation of MP-supported
    Keras layers in order to avoid merging an already parallelised layer

    In case that the input tensor is a Run:AI parallelised layer, we duplicate
    this layer and activate it on every part of the parallelised input layer.
    Otherwise, the layer is activated on the non-parallelised input layer,
    just as it was supposed to run originally, and the input tensor is not
    being split. Anyways, the layer actions and algorithm are kept untouched.

    If the input layer is indeed a Run:AI parallelised one, the merged tensor
    is the concatenation of the layer's outputs. The channel dimension is taken
    from the 'data_format' attribute, if one exists, otherwise it is assumed to
    be the last dimension.

    The following example shows the essence of the inheritance being done, where:
        1) 'Layer' is keras.layers.Layer
        2) 'Keras' is the Keras implementation of a layer (e.g. keras.layers.Dropout)
        3) 'Runai' is this very class (runai.mp.layers.Keep)
        4) 'Final' is the MP-supported Keras layer implementation

    class Layer(object):
        def call(self):
            print('Layer.call()')

    class Keras(Layer):
        def call(self):
            print('Keras.call()')
            super(Keras, self).call()

    class Runai(Layer):
        def call(self):
            print('Runai.call()')
            super(Runai, self).call()

    class Final(Runai, Keras): pass
    """
    def call(self, inputs):
        if isinstance(inputs, (tuple, list)):
            valid = all([coordinator.registered(input) for input in inputs])
        else:
            valid = coordinator.registered(inputs)

        if not valid:
            log.debug('Not parallelising \'%s\' layer "%s"', self.__class__.__name__, getattr(self, 'name', 'N/A'))
            return super(Keep, self).call(inputs)

        log.info('Using parallelised input for \'%s\' layer "%s"', self.__class__.__name__, getattr(self, 'name', 'N/A'))

        channel_axis = 1 if getattr(self, 'data_format', 'channels_last') == 'channels_first' else -1

        def output(gpu, inputs):
            with tf.device('/device:GPU:%d' % gpu):
                return super(Keep, self).call(inputs)

        if isinstance(inputs, (tuple, list)):
            inputs = [coordinator.resolve(input) for input in inputs]
            inputs = [[input[i] for input in inputs] for i in range(len(inputs[0]))]
        else:
            inputs = coordinator.resolve(inputs)
        
        outputs = [output(gpu, input) for gpu, input in enumerate(inputs)]
        merged = keras.layers.Concatenate(axis=channel_axis)(outputs)
        coordinator.register(merged, outputs)
        return merged
