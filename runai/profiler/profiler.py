import json
import os

from tensorflow.python.client import timeline

from runai.utils import Hook

class Profiler(Hook):
    def __init__(self, module, method, steps, dst):
        super(Profiler, self).__init__(module, method)
        self._timeline = None
        self._step = 0
        self._steps = steps
        self._dst = dst
    
    def _update(self, run_metadata):
        chrome_trace = json.loads(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())
        
        if self._timeline is None:
            self._timeline = chrome_trace
        else:
            self._timeline['traceEvents'] += [event for event in chrome_trace['traceEvents'] if 'ts' in event]

        if self._step % self._steps == 0:
            with open(os.path.join(self._dst, 'timeline_%d' % self._step), 'w') as f:
                f.write(json.dumps(self._timeline))

        self._step += 1
