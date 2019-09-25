import keras.optimizers

from runapy import log
from runapy.ga.keras.optimizers import Optimizer as GA

def compile(self, *args, **kwargs):
    if 'optimizer' in kwargs:
        assert len(args) == 0
        optimizer = kwargs['optimizer']
    else:
        assert len(args) == 1
        optimizer = args[0]

    # TODO(levosos): support cases when 'optimizer' is not an class instance but is either a string or a dictionary
    if not isinstance(optimizer, keras.optimizers.Optimizer):
        raise ValueError("'optimizer' must be a valid keras.optimizers.Optimizer")

    log.debug('compile() called with optimizer %s', optimizer)
    
    config = self.__runai__['config']

    if config.steps > 1:
        log.debug('Using GA with %d steps', config.steps)
        optimizer = GA(optimizer, config.steps)
    else:
        log.debug('GA is not needed')

    kwargs['optimizer'] = optimizer

    return self.__runai__['compile'](**kwargs) # ignore 'args' as 'optimizer' is the only possible argument and it is in 'kwargs'

def fit(self, *args, **kwargs):
    log.debug('fit() called')
    return self.__runai__['fit'](*args, **kwargs)

def fit_generator(self, *args, **kwargs):
    log.critical('fit_generator() called')
    raise NotImplementedError('fit_generator() is not yet supported with Run:AI')

def Model(model, config):
    __runai__ = dict(
        config=config,
    )

    for method in [compile, fit, fit_generator]: # TODO(levosos): what about evaluate() and predict()?
        __runai__[method.__name__] = getattr(model, method.__name__)
        setattr(model, method.__name__, method.__get__(model))
    
    setattr(model, '__runai__', __runai__)
    
    return model
