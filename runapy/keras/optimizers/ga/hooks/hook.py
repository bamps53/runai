class Hook(object):
    def __init__(self, module, method):
        self.module   = module
        self.method   = method
        self.original = getattr(module, method)

    def __enter__(self):
        setattr(self.module, self.method, self.impl)

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(self.module, self.method, self.original)
    
    def impl(self, *args, **kwargs):
        raise NotImplementedError()
