from __future__ import absolute_import # to support 'import keras.backend' in Python 2

import enum

class Method(enum.Enum):
    Cin  = 0
    Cout = 1

from . import keras

import runai.utils

def init(splits, method):
    if method != Method.Cin and method != Method.Cout:
        raise ValueError('Unrecognized method: %s' % method)

    runai.utils.log.info('Initializing MP (%s) with %s splits', method.name, splits)
    
    import sys

    setattr(sys.modules[__name__], 'splits', splits)
    setattr(sys.modules[__name__], 'method', method)

    import keras.layers

    from .keras import layers

    [setattr(keras.layers, attribute, getattr(layers, attribute)) for attribute in [
        'Activation',
        'add',
        'Add',
        'average',
        'Average',
        'AveragePooling2D',
        'BatchNormalization',
        'Conv2D',
        'Dense',
        'Dropout',
        'Flatten',
        'GlobalAveragePooling2D',
        'GlobalMaxPooling2D',
        'maximum',
        'Maximum',
        'MaxPooling2D',
        'minimum',
        'Minimum',
        'multiply',
        'Multiply',
        'subtract',
        'Subtract',
        'ZeroPadding2D',
        ]]
