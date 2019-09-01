import keras.optimizers
import keras.backend as K

from . import hooks

class Optimizer(keras.optimizers.Optimizer):
    def __init__(self, optimizer, steps, **kwargs):
        super(Optimizer, self).__init__()
        self.optimizer = optimizer
        self.steps     = steps

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        with K.name_scope("runai") as name_scope:
            iterations = K.variable(0, dtype='int64', name='iterations')
            first      = K.equal(iterations % self.steps, 0)
            last       = K.equal(iterations % self.steps, self.steps - 1)

            agrads = [K.zeros(K.int_shape(param), dtype=K.dtype(param)) for param in params]

            for grad, agrad in zip(grads, agrads):
                self.updates.append(
                    K.update(agrad, K.switch(first, grad, agrad + grad))
                )

        with hooks.get_gradients(self.optimizer, agrads),               \
            hooks.update    (condition=last, name_scope=name_scope),    \
            hooks.update_add(condition=last, name_scope=name_scope),    \
            hooks.update_sub(condition=last, name_scope=name_scope):
            for update in self.optimizer.get_updates(loss, params):
                # get_updates() may return assignment ops or tuples (variable, gradient) representing the desired assignments
                # in the latter case we want to build the assignment op ourselves under the same hooks
                #
                # reference: https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/keras/backend.py#L3145
                if isinstance(update, tuple):
                    with K.name_scope(name_scope):
                        update = K.update(update[0], update[1])

                self.updates.append(update)

        assert K.backend() == 'tensorflow', "Unsupported backend (" + K.backend() + ")"

        with K.name_scope(name_scope), K.get_session().graph.control_dependencies(self.updates):
            self.updates.append(K.update_add(iterations, 1))
        
        return self.updates

    def get_gradients(self, loss, params):
        return self.optimizer.get_gradients(loss, params)

    def set_weights(self, weights):
        self.optimizer.set_weights(weights)

    def get_weights(self):
        return self.optimizer.get_weights()
    
    def get_config(self):
        return self.optimizer.get_config()
