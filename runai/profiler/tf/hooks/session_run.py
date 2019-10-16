from runai.profiler import Profiler

import tensorflow

class session_run(Profiler):
    def __init__(self, steps, dst):
        super(session_run, self).__init__(tensorflow.Session, 'run', steps, dst)
    
    def __hook__(self, _self, fetches, feed_dict=None, options=None, run_metadata=None):
        assert options is None # TODO(levosos): handle this
        assert run_metadata is None # TODO(levosos): handle this

        options = tensorflow.RunOptions(trace_level=tensorflow.RunOptions.FULL_TRACE)
        run_metadata = tensorflow.RunMetadata()

        result = self.__original__(_self, fetches, feed_dict, options, run_metadata)

        self._update(run_metadata)

        return result
