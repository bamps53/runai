import random
import sys
import unittest

import runai.utils

def foo():
    return 42

class Object:
    def foo(self):
        return 42

class Hook(runai.utils.Hook):
    def __init__(self, module):
        super(Hook, self).__init__(module, 'foo')

    def impl(self):
        return 217

class TestHook(unittest.TestCase):
    def test_module_scope(self):
        assert foo() == 42

        with Hook(sys.modules[__name__]):
            assert foo() == 217

        assert foo() == 42

    def test_module_manual(self):
        hook = Hook(sys.modules[__name__])
        assert foo() == 42

        hook.enable()
        assert foo() == 217

        hook.disable()
        assert foo() == 42

    def test_object(self):
        o = Object()
        
        assert o.foo() == 42

        with Hook(o):
            assert o.foo() == 217

        assert o.foo() == 42

    def test_object_manual(self):
        o = Object()
        hook = Hook(o)
        assert o.foo() == 42
        
        hook.enable()
        assert o.foo() == 217
        
        hook.disable()
        assert o.foo() == 42

class TestRandom(unittest.TestCase):
    def test_string(self):
        for _ in range(100):
            length = random.randint(1, 10)
            a = runai.utils.random.string(length=length)
            b = runai.utils.random.string(length=length)

            assert len(a) == length
            assert len(b) == length

            assert a != b

if __name__ == '__main__':
    unittest.main()
