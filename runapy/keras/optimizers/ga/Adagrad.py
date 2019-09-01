import keras.optimizers

from .optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(self, steps, **kwargs):
        super(Adagrad, self).__init__(
            optimizer=keras.optimizers.Adagrad(**kwargs),
            steps=steps,
            **kwargs
        )
