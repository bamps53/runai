import unittest
import random

import keras.backend as K
import keras.optimizers
import numpy as np

import runapy.keras.optimizers.ga

class Optimizer(unittest.TestCase):
    def _run(self, params, gradients, optimizer):
        params = [K.variable(param) for param in params]
        placeholders = [K.placeholder(shape=params[i].shape) for i in range(len(params))]

        with runapy.keras.optimizers.ga.hooks.get_gradients(optimizer, placeholders):
            updates = optimizer.get_updates(None, params) # 'loss' argument is unnecessary because of our hook
        
        for step in range(len(gradients)):
            K.get_session().run(
                updates,
                feed_dict={ placeholders[i]: gradients[step][i] for i in range(len(params)) }
            )
        
        return K.get_session().run(params)
    
    def _test(self, optimizer):
        count = random.randint(2, 10)
        steps = random.randint(2, 10)
        shape = [random.randint(2, 5) for _ in range(2, 5)]

        # numpy generates float64 values by default, whereas TF uses float32 by default
        # we explictly generate float32 values because passing 64-bit floats to TF causes floating-point issues
        params = [np.random.random(shape).astype(np.float32) for _ in range(count)]
        gradients = [[np.random.random(shape).astype(np.float32) for _ in range(count)] for _ in range(steps)]

        us = self._run(
            params,
            gradients,
            getattr(runapy.keras.optimizers.ga, optimizer)(steps=steps)
        )
        
        them = self._run(
            params,
            [[np.array([gradients[step][i] for step in range(steps)]).sum(axis=0) for i in range(count)]], # running a single step with the reduced (sum) gradients
            getattr(keras.optimizers, optimizer)()
        )

        self.assertTrue(np.array_equal(us, them))
    
    def testAdadelta(self):
        self._test('Adadelta')

    def testAdagrad(self):
        self._test('Adagrad')

    def testAdam(self):
        self._test('Adam')

    def testAdamax(self):
        self._test('Adamax')

    def testNadam(self):
        self._test('Nadam')

    def testRMSprop(self):
        self._test('RMSprop')

    def testSGD(self):
        self._test('SGD')

if __name__ == '__main__':
    unittest.main()
