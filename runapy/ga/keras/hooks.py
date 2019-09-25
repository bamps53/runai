import keras.backend as K

from runapy.utils import Hook

class get_gradients(Hook):
    def __init__(self, optimizer, gradients):
        super(get_gradients, self).__init__(optimizer, 'get_gradients')
        self.gradients = gradients
    
    def impl(self, loss, params):
        return self.gradients

class update_add(Hook):
    def __init__(self, condition, name_scope):
        super(update_add, self).__init__(K, 'update_add')
        self.condition = condition
        self.name_scope = name_scope
    
    def impl(self, x, increment):
        with K.name_scope(self.name_scope):
            if not K.is_tensor(increment):
                increment = K.constant(increment, dtype=K.dtype(x))

            increment = K.switch(self.condition, increment, K.constant(0, dtype=K.dtype(x)))

        return self.original(x, increment)

class update_sub(Hook):
    def __init__(self, condition, name_scope):
        super(update_sub, self).__init__(K, 'update_sub')
        self.condition = condition
        self.name_scope = name_scope
    
    def impl(self, x, decrement):
        with K.name_scope(self.name_scope):
            if not K.is_tensor(decrement):
                decrement = K.constant(decrement, dtype=K.dtype(x))

            decrement = K.switch(self.condition, decrement, K.constant(0, dtype=K.dtype(x)))

        return self.original(x, decrement)

class update(Hook):
    def __init__(self, condition, name_scope):
        super(update, self).__init__(K, 'update')
        self.condition = condition
        self.name_scope = name_scope
    
    def impl(self, x, new_x):
        with K.name_scope(self.name_scope):
            if not K.is_tensor(new_x):
                new_x = K.constant(new_x, dtype=K.dtype(x))

            new_x = K.switch(self.condition, new_x, x)

        return self.original(x, new_x)
