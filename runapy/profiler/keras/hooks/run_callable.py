import json
import os

import tensorflow
from tensorflow.python import pywrap_tensorflow as tf_c_api
from tensorflow.python.client import timeline
from tensorflow.python.util import compat

from runapy.utils import Hook

class run_callable(Hook):
    def __init__(self, steps, dst):
        super(run_callable, self).__init__(tf_c_api, 'TF_SessionRunCallable')
        self._timeline = None
        self._step = 0
        self._steps = steps
        self._dst = dst
    
    def impl(self, session, handle, args, status, run_metadata_ptr):
        assert run_metadata_ptr is None # TODO(levosos): handle this

        run_metadata_ptr = tf_c_api.TF_NewBuffer()
        result = self.original(session, handle, args, status, run_metadata_ptr)

        proto = tf_c_api.TF_GetBuffer(run_metadata_ptr)
        tf_c_api.TF_DeleteBuffer(run_metadata_ptr)
        
        run_metadata = tensorflow.RunMetadata()
        run_metadata.ParseFromString(compat.as_bytes(proto))

        chrome_trace = json.loads(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())
        
        if self._timeline is None:
            self._timeline = chrome_trace
        else:
            self._timeline['traceEvents'] += [event for event in chrome_trace['traceEvents'] if 'ts' in event]

        if self._step % self._steps == 0:
            with open(os.path.join(self._dst, 'timeline_%d' % self._step), 'w') as f:
                f.write(json.dumps(self._timeline))

        self._step += 1

        return result
