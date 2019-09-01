from .hook import Hook

import keras.backend as K

class update(Hook):
    def __init__(self, condition, name_scope):
        super(update, self).__init__(K, 'update')
        self.condition  = condition
        self.name_scope = name_scope
    
    def impl(self, x, new_x):
        with K.name_scope(self.name_scope):
            if not K.is_tensor(new_x):
                new_x = K.constant(new_x, dtype=K.dtype(x))

            new_x = K.switch(self.condition, new_x, x)

        return self.original(x, new_x)
