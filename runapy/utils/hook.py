class Hook(object):
    def __init__(self, module, method):
        self.module   = module
        self.method   = method
        self.original = getattr(module, method)

    def replace(self, method):
        assert getattr(self.module, self.method) != method
        setattr(self.module, self.method, method)

    def enable(self):
        self.replace(self.impl)

    def disable(self):
        self.replace(self.original)

    def __enter__(self):
        self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()
    
    def impl(self):
        raise NotImplementedError()
