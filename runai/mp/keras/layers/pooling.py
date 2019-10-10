import keras.layers

from runai import log

from . import coordinator

class MaxPooling2D(keras.layers.MaxPooling2D):
    def call(self, inputs):
        assert not isinstance(inputs, (tuple, list))

        if not coordinator.registered(inputs):
            log.debug('Not parallelising \'%s\' layer "%s"', self.__class__.__name__, self.name)
            return super(MaxPooling2D, self).call(inputs)

        log.info('Using parallelised input for \'%s\' layer "%s"', self.__class__.__name__, self.name)

        channel_axis = 1 if self.data_format == 'channels_first' else -1

        inputs = coordinator.resolve(inputs)
        outputs = [super(MaxPooling2D, self).call(input) for input in inputs]
        merged = keras.layers.Concatenate(axis=channel_axis)(outputs)
        coordinator.register(merged, outputs)
        return merged
