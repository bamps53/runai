import os

# NOTE(levosos): we do not use TensorFlow's API (tensorflow.python.client.device_lib.list_local_devices())
# because it causes all devices to be mapped to the current process and when using Horovod
# we receive the next error: "TensorFlow device (GPU:0) is being mapped to multiple CUDA devices".

DIR = "/proc/driver/nvidia/gpus"

def available():
    return os.path.isdir(DIR)

def count():
    if not available():
        return 0
    
    return len(os.listdir(DIR))
