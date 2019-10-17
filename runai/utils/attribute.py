def rename(obj, old, new):
    setattr(obj, new, getattr(obj, old))
    delattr(obj, old)

class Attribute(object):
    """ A utility class for creating temporary attributes in objects
    """

    def __init__(self, obj, name, value):
        assert isinstance(name, list) == isinstance(value, list)

        self.obj = obj
        self.name = name
        self.value = value

    def __enter__(self):
        if isinstance(self.name, list):
            for name, value in zip(self.name, self.value):
                setattr(self.obj, name, value)
        else:
            setattr(self.obj, self.name, self.value)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.name, list):
            for name in self.name:
                delattr(self.obj, name)
        else:
            delattr(self.obj, self.name)
