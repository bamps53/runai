import keras.optimizers

from optimizer import Optimizer

class Adadelta(Optimizer):
    def __init__(self, steps, **kwargs):
        super(Adadelta, self).__init__(
            optimizer=keras.optimizers.Adadelta(**kwargs),
            steps=steps,
            **kwargs
        )
