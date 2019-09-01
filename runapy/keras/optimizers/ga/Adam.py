import keras.optimizers

from optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, steps, **kwargs):
        super(Adam, self).__init__(
            optimizer=keras.optimizers.Adam(**kwargs),
            steps=steps,
            **kwargs
        )
