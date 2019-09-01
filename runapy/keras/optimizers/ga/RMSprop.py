import keras.optimizers

from .optimizer import Optimizer

class RMSprop(Optimizer):
    def __init__(self, steps, **kwargs):
        super(RMSprop, self).__init__(
            optimizer=keras.optimizers.RMSprop(**kwargs),
            steps=steps,
            **kwargs
        )
