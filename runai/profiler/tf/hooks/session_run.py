from runai.profiler import Profiler

import tensorflow

class session_run(Profiler):
    def __init__(self, steps, dst):
        super(session_run, self).__init__(tensorflow.Session, 'run', steps, dst)
    
    def __hook__(self, _self, fetches, feed_dict=None, options=None, run_metadata=None):
        if options is None:
            options = tensorflow.RunOptions()
    
        options.trace_level = tensorflow.RunOptions.FULL_TRACE

        if run_metadata is None:
            run_metadata = tensorflow.RunMetadata()

        result = self.__original__(_self, fetches, feed_dict, options, run_metadata)

        self._update(run_metadata)

        return result
