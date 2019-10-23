import sys

from .keep import Keep

[setattr(sys.modules[__name__], layer, Keep.create(layer)) for layer in [
    'MaxPooling2D',
    'AveragePooling2D',
    'GlobalAveragePooling2D',
    'GlobalMaxPooling2D',
]]
