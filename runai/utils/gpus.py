# NOTE(levosos): we do not use TensorFlow's API (tensorflow.python.client.device_lib.list_local_devices())
# because it causes all devices to be mapped to the current process and when using Horovod
# we receive the next error: "TensorFlow device (GPU:0) is being mapped to multiple CUDA devices".

import subprocess

def available():
    return count() > 0

def count():
    try:
        return int(subprocess.check_output('nvidia-smi --list-gpus | wc -l', shell=True))
    except subprocess.CalledProcessError:
        return 0
