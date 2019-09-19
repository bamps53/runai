from runapy.profiler.keras import hooks

def profile(steps, dst='/tmp/'):
    profile.make_callable = hooks.make_callable()
    profile.run_callable = hooks.run_callable(steps, dst)

    profile.make_callable.enable()
    profile.run_callable.enable()

