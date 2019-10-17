import keras.layers

from .keep import Keep

class Add(Keep, keras.layers.Add): pass

class Subtract(Keep, keras.layers.Subtract): pass

class Multiply(Keep, keras.layers.Multiply): pass

class Average(Keep, keras.layers.Average): pass

class Maximum(Keep, keras.layers.Maximum): pass

class Minimum(Keep, keras.layers.Minimum): pass
