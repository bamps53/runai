import keras.optimizers

from optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, steps, **kwargs):
        super(SGD, self).__init__(
            optimizer=keras.optimizers.SGD(**kwargs),
            steps=steps,
            **kwargs
        )
