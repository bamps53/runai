from runapy.utils import Hook

import keras.backend as K

class update_sub(Hook):
    def __init__(self, condition, name_scope):
        super(update_sub, self).__init__(K, 'update_sub')
        self.condition  = condition
        self.name_scope = name_scope
    
    def impl(self, x, decrement):
        with K.name_scope(self.name_scope):
            if not K.is_tensor(decrement):
                decrement = K.constant(decrement, dtype=K.dtype(x))

            decrement = K.switch(self.condition, decrement, K.constant(0, dtype=K.dtype(x)))

        return self.original(x, decrement)
