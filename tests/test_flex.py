import random
import unittest

import runapy.flex

class Init(unittest.TestCase):
    def testStatics(self):
        global_batch_size = random.randint(10, 100)
        max_gpu_batch_size = random.randint(1, global_batch_size) # must be less than 'global_batch_size' because we don't support uneven spread
        gpus = 1 # currently multi-gpu is not supported in tests
        
        runapy.flex.init(
            global_batch_size=global_batch_size,
            max_gpu_batch_size=max_gpu_batch_size,
            gpus=gpus
        )

        self.assertEqual(runapy.flex.global_batch_size, global_batch_size)
        self.assertEqual(runapy.flex.gpus, gpus)
        self.assertEqual(runapy.flex.master, True)
        
        self.assertFalse(hasattr(runapy.flex, 'hvd'))

if __name__ == '__main__':
    unittest.main()
