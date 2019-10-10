import keras.layers

from runai import log

from . import coordinator

class Dropout(keras.layers.Dropout):
    def call(self, inputs):
        assert not isinstance(inputs, (tuple, list))

        if not coordinator.registered(inputs):
            log.debug('Not parallelising dropout layer \'%s\'', self.name)
            return super(Dropout, self).call(inputs)

        log.info('Using parallelised input for dropout layer \'%s\'', self.name)

        inputs = coordinator.resolve(inputs)
        outputs = [super(Dropout, self).call(input) for input in inputs]
        merged = keras.layers.Concatenate(axis=-1)(outputs)
        coordinator.register(merged, outputs)
        return merged
