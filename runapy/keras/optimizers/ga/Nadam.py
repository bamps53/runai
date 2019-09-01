import keras.optimizers

from .optimizer import Optimizer

class Nadam(Optimizer):
    def __init__(self, steps, **kwargs):
        super(Nadam, self).__init__(
            optimizer=keras.optimizers.Nadam(**kwargs),
            steps=steps,
            **kwargs
        )
