import tensorflow

from .profiler import Profiler
from . import keras
from . import tf

def profile(steps, dst='/tmp/'):
    profile.session_run = tf.hooks.session_run(steps, dst)
    profile.session_run.enable()

    if '_Callable' in dir(tensorflow.Session): # Keras new API
        profile.make_callable = keras.hooks.make_callable()
        profile.run_callable = keras.hooks.run_callable(steps, dst)

        profile.make_callable.enable()
        profile.run_callable.enable()
