from __future__ import absolute_import # to support 'import keras.backend' in Python 2

import enum

class Method(enum.Enum):
    Cin  = 0
    Cout = 1

from . import keras

from runai.utils import log

def init(splits, method):
    if method != Method.Cin and method != Method.Cout:
        raise ValueError('Unrecognized method: %s' % method)

    log.info('Initializing MP (%s) with %s splits', method.name, splits)
    
    import sys

    setattr(sys.modules[__name__], 'splits', splits)
    setattr(sys.modules[__name__], 'method', method)

    import keras.layers

    from .keras import layers

    def _layer(layer):
        setattr(
            keras.layers,
            layer,
            getattr(layers, layer)
        )
    
    [_layer(layer) for layer in [
        'Activation',
        'Add',
        'Average',
        'BatchNormalization',
        'Conv2D',
        'Dense',
        'Dropout',
        'Maximum',
        'MaxPooling2D',
        'Minimum',
        'Multiply',
        'Subtract',
        'ZeroPadding2D',
        ]]

    def _method(method):
        setattr(
            keras.layers,
            method,
            lambda inputs, *args, **kwargs: getattr(layers, method.capitalize())(*args, **kwargs)(inputs)
        )

    [_method(method) for method in ['add', 'subtract', 'multiply', 'average', 'maximum', 'minimum']]
