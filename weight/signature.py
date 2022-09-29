import joblib
import sys

from questions.memory import memory


@memory.cache(ignore=['clausifier', 'parallel'])
def get_signatures(problem_names, clausifier, parallel=None):
    if parallel is None:
        parallel = joblib.Parallel()

    def get_signature(problem):
        try:
            # Raises `RuntimeError` when clausification fails
            return clausifier.signature(problem)
        except RuntimeError:
            return None

    print(f'Collecting signatures of {len(problem_names)} problems', file=sys.stderr)
    return parallel(joblib.delayed(get_signature)(problem) for problem in problem_names)
