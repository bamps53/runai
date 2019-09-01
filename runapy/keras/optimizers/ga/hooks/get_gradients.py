from hook import Hook

class get_gradients(Hook):
    def __init__(self, optimizer, gradients):
        super(get_gradients, self).__init__(optimizer, 'get_gradients')
        self.gradients = gradients
    
    def impl(self, loss, params):
        return self.gradients
