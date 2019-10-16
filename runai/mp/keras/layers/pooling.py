import keras.layers

from .keep import Keep

class MaxPooling2D(Keep, keras.layers.MaxPooling2D): pass
