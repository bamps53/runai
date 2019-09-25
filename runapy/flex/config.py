from runapy import log

class Config:
    """Config

    # Arguments
        lr: float >= 0. Total learning rate.
        global_batch_size: integer > 0. Global effective batch size per step.
        max_gpu_batch_size: integer > 0. Maximum batch size that fits in a GPU.
        gpus: integer > 0. Number of available GPUs
    """

    def __init__(self, lr, global_batch_size, max_gpu_batch_size, gpus):
        self.lr = lr
        self.global_batch_size = global_batch_size
        self.gpus = gpus
        self.master = True
        
        # TODO(levosos): support uneven dividing
        self.steps = global_batch_size // (max_gpu_batch_size * gpus)
        self.batch_size = global_batch_size // (self.steps * gpus)

        log.info('Spreading global batch size %d accross %d GPU(s) each with %d step(s) of batch size %d',
            global_batch_size, gpus, self.steps, self.batch_size)

        if gpus != 1:
            raise ValueError('Multi-GPU is still not supported with Run:AI')
