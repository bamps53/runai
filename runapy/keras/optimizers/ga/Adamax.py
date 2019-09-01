import keras.optimizers

from .optimizer import Optimizer

class Adamax(Optimizer):
    def __init__(self, steps, **kwargs):
        super(Adamax, self).__init__(
            optimizer=keras.optimizers.Adamax(**kwargs),
            steps=steps,
            **kwargs
        )
