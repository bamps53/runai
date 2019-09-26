import sys

import keras.optimizers
import keras.backend as K

from runai import log

from . import hooks

# NOTE(levosos): this is very much inspired by Horovod (create_distributed_optimizer() @ horovod/_keras/__init__.py)
def Optimizer(optimizer, steps):
    """
    Wraps any valid Keras optimizer with gradient accumulation
    """
    class _GradientAccumulationOptimizer(keras.optimizers.Optimizer):
        def __init__(self, steps, **kwargs):
            self.__class__.__optimizer__.__init__(self, **kwargs)
            self.steps = steps

            log.debug('Wrapping \'%s\' Keras optimizer with GA of %d steps', self.__class__.__base__.__name__, steps)

        def get_updates(self, loss, params):
            grads = self.get_gradients(loss, params)

            with K.name_scope("runai") as name_scope:
                iterations = K.variable(0, dtype='int64', name='iterations')
                first      = K.equal(iterations % self.steps, 0)
                last       = K.equal(iterations % self.steps, self.steps - 1)

                # variables to hold the accumulated gradients between steps
                vagrads = [K.zeros(K.int_shape(param), dtype=K.dtype(param)) for param in params]
                
                # reset the accumulated gradient every first iteration
                agrads = [K.switch(first, grad, grad + vagrad) for grad, vagrad in zip(grads, vagrads)]

            with hooks.get_gradients(self, [agrad / self.steps for agrad in agrads]), \
                hooks.update    (condition=last, name_scope=name_scope),    \
                hooks.update_add(condition=last, name_scope=name_scope),    \
                hooks.update_sub(condition=last, name_scope=name_scope):

                def build(update):
                    # get_updates() may return assignment ops or tuples (variable, gradient) representing the desired assignments
                    # in the latter case we want to build the assignment op ourselves under the same hooks
                    #
                    # reference: https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/keras/backend.py#L3145
                    if isinstance(update, tuple):
                        with K.name_scope(name_scope):
                            return K.update(update[0], update[1])
                    
                    return update
                    
                self.updates = [build(update) for update in self.__class__.__optimizer__.get_updates(self, loss, params)]

            assert K.backend() == 'tensorflow', "Unsupported backend (" + K.backend() + ")"

            with K.name_scope(name_scope):
                self.updates.extend([K.update(vagrad, agrad) for vagrad, agrad in zip(vagrads, agrads)])

                with K.get_session().graph.control_dependencies(self.updates):
                    self.updates.append(K.update_add(iterations, 1))
            
            return self.updates

        def get_config(self):
            # we have to support creating our optimizers from configurations in order to support being run with Horovod
            # Horovod dynamically creates a class that inherits the optimizer class it's wrapping (our optimizers), and
            # passes the dictionary returned from this very method as the kwargs for the initialization in __init__()
            config = self.__class__.__optimizer__.get_config(self)
            config['steps'] = self.steps
            return config
    
    # the main idea is to dynamically create a class that has all the functionality of the passed optimizer
    # (this is done by inheriting it) while overriding get_updates() to accumulate the gradients and actually
    # assign them once in a few steps
    d = dict(_GradientAccumulationOptimizer.__dict__)
    d['__optimizer__'] = optimizer.__class__

    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        d
    )
    return cls(steps, **optimizer.get_config())

def _optimizer(optimizer):
    setattr(
        sys.modules[__name__],
        optimizer,
        lambda steps, **kwargs: Optimizer(optimizer=getattr(keras.optimizers, optimizer)(**kwargs), steps=steps)
    )

[_optimizer(optimizer) for optimizer in ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']]
