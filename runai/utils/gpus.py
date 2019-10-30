from tensorflow.python.client import device_lib

def count():
    return len([device for device in device_lib.list_local_devices() if device.device_type == 'GPU'])
