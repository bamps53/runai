from runapy.utils import Hook

import keras.backend as K

class update_add(Hook):
    def __init__(self, condition, name_scope):
        super(update_add, self).__init__(K, 'update_add')
        self.condition  = condition
        self.name_scope = name_scope
    
    def impl(self, x, increment):
        with K.name_scope(self.name_scope):
            if not K.is_tensor(increment):
                increment = K.constant(increment, dtype=K.dtype(x))

            increment = K.switch(self.condition, increment, K.constant(0, dtype=K.dtype(x)))

        return self.original(x, increment)
