import random
import unittest

import runai.flex

class Init(unittest.TestCase):
    def testStatics(self):
        global_batch_size = random.randint(10, 100)
        max_gpu_batch_size = random.randint(1, global_batch_size) # must be less than 'global_batch_size' because we don't support uneven spread
        gpus = 1 # currently multi-gpu is not supported in tests
        
        runai.flex.init(
            global_batch_size=global_batch_size,
            max_gpu_batch_size=max_gpu_batch_size,
            gpus=gpus
        )

        self.assertEqual(runai.flex.global_batch_size, global_batch_size)
        self.assertEqual(runai.flex.gpus, gpus)
        self.assertEqual(runai.flex.master, True)
        
        self.assertFalse(hasattr(runai.flex, 'hvd'))

if __name__ == '__main__':
    unittest.main()
