import tempfile

import benchexec.container
import benchexec.runexecutor


def run(args, **kwargs):
    dir_modes = {
        '/': benchexec.container.DIR_READ_ONLY,
        '/run': benchexec.container.DIR_HIDDEN,
        '/tmp': benchexec.container.DIR_HIDDEN
    }
    executor = benchexec.runexecutor.RunExecutor(dir_modes=dir_modes)
    with tempfile.NamedTemporaryFile() as output_file:
        result = executor.execute_run(args=args, output_filename=output_file.name, **kwargs)
        output_file.seek(0)
        output = output_file.read().decode()
    try:
        terminationreason = result['terminationreason']
    except KeyError:
        terminationreason = None
    return {
        'args': args,
        'backend_kwargs': kwargs,
        'terminationreason': terminationreason,
        'stdout': output,
        'meta': result
    }
