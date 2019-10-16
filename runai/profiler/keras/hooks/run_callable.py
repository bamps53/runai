from runai.profiler import Profiler

import tensorflow
from tensorflow.python import pywrap_tensorflow as tf_c_api
from tensorflow.python.util import compat

class run_callable(Profiler):
    def __init__(self, steps, dst):
        super(run_callable, self).__init__(tf_c_api, 'TF_SessionRunCallable', steps, dst)
    
    def __hook__(self, session, handle, args, status, run_metadata_ptr):
        assert run_metadata_ptr is None # TODO(levosos): handle this

        run_metadata_ptr = tf_c_api.TF_NewBuffer()
        result = self.__original__(session, handle, args, status, run_metadata_ptr)

        proto = tf_c_api.TF_GetBuffer(run_metadata_ptr)
        tf_c_api.TF_DeleteBuffer(run_metadata_ptr)
        
        run_metadata = tensorflow.RunMetadata()
        run_metadata.ParseFromString(compat.as_bytes(proto))

        self._update(run_metadata)

        return result
