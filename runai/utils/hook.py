class Hook(object):
    def __init__(self, module, method):
        self.module   = module
        self.method   = method
        self.original = getattr(module, method)

    def enable(self):
        setattr(self.module, self.method, lambda *args, **kwargs: self.impl(*args, **kwargs))

    def disable(self):
        setattr(self.module, self.method, self.original)

    def __enter__(self):
        self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()
    
    def impl(self):
        raise NotImplementedError()
