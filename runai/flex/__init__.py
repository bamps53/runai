from __future__ import absolute_import # to support 'import keras.backend' in Python 2

import sys

from runai import log

from . import keras

def init(global_batch_size, max_gpu_batch_size, gpus):
    if gpus < 1:
        raise ValueError('GPU count (%d) must be at least 1' % gpus)

    module = sys.modules[__name__]

    setattr(module, 'global_batch_size', global_batch_size)
    setattr(module, 'gpus', gpus)
    setattr(module, 'master', True)

    # TODO(levosos): support uneven dividing
    steps = global_batch_size // (max_gpu_batch_size * gpus)
    batch_size = global_batch_size // (steps * gpus)
    
    setattr(module, 'steps', steps)
    setattr(module, 'batch_size', batch_size)

    log.info('Spreading global batch size %d across %d GPU(s) each with %d step(s) of batch size %d',
        global_batch_size, gpus, steps, batch_size)

    if gpus > 1:
        log.debug('Initializing Horovod')
        import horovod.keras as hvd
        hvd.init()
        setattr(module, 'master', hvd.local_rank() == 0)
        setattr(module, 'hvd', hvd) # so that anyone will be easily accessible to Horovod
        
        log.debug('Attaching Keras session to GPU #%d', hvd.local_rank())
        import tensorflow
        config = tensorflow.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        import keras.backend
        keras.backend.set_session(tensorflow.Session(config=config)) # TODO(levosos): support cases where configuration will be set afterwards
