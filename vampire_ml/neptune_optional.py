import neptune

initialized = False


def init(*args, **kwargs):
    neptune.init(*args, **kwargs)
    global initialized
    initialized = True


def create_experiment(*args, **kwargs):
    if initialized:
        neptune.create_experiment(*args, **kwargs)


def set_property(*args, **kwargs):
    if initialized:
        neptune.set_property(*args, **kwargs)


def log_artifact(*args, **kwargs):
    if initialized:
        neptune.log_artifact(*args, **kwargs)
