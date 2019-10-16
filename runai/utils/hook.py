class Hook(object):
    def __init__(self, module, method):
        self.module = module
        self.method = method
        self.__original__ = getattr(module, method)

    def enable(self):
        setattr(self.module, self.method, lambda *args, **kwargs: self.__hook__(*args, **kwargs))

    def disable(self):
        setattr(self.module, self.method, self.__original__)

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()
    
    def __hook__(self):
        raise NotImplementedError()
