__mapping__ = {}

def register(merged, outputs):
    __mapping__[merged] = outputs

def registered(merged):
    return merged in __mapping__

def resolve(merged):
    return __mapping__[merged]
