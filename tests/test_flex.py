import random
import unittest

import runapy.flex

setattr(random, 'lr', lambda: random.random())
setattr(random, 'batch_size', lambda max=None: random.randint(1, max if max else 100))
setattr(random, 'gpus', lambda: 1) # currently multi-gpu is not supported

class Config(unittest.TestCase):
    def testStatics(self):
        lr = random.lr()
        global_batch_size = random.batch_size()
        gpus = random.gpus()
        
        config = runapy.flex.Config(
            lr=lr,
            global_batch_size=global_batch_size,
            max_gpu_batch_size=random.batch_size(global_batch_size), # must be less than 'global_batch_size' because we don't support uneven spread
            gpus=gpus
        )

        self.assertEqual(config.lr, lr)
        self.assertEqual(config.global_batch_size, global_batch_size)
        self.assertEqual(config.gpus, gpus)

if __name__ == '__main__':
    unittest.main()
