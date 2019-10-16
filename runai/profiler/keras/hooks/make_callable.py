import tensorflow

from runai.utils import Hook

class make_callable(Hook):
    def __init__(self):
        super(make_callable, self).__init__(tensorflow.Session, '_make_callable_from_options')
    
    def __hook__(self, _self, callable_options):
        assert not callable_options.HasField('run_options') # TODO(levosos): support this

        # TODO(levosos): should we create a copy of 'callable_options'?
        callable_options.run_options.trace_level = tensorflow.RunOptions.FULL_TRACE

        return self.__original__(_self, callable_options)
