import hashlib
import json
import logging
import os

from . import problems
from . import questions


def get_datasets(problem_patterns, validation_split, max_problems, questions_dir, batch_size, cache_dir=None,
                 cache_mem=False):
    # We need to split problems first and then collect questions for each of the datasets
    # because not all problems have questions and we only generate questions samples
    # for problems with at least one question.
    prob = problems.get_datasets_split(problem_patterns, validation_split, max_problems)

    def get_question_batches_dataset(p, k):
        q = questions.individual.get_dataset(questions_dir, p)
        if cache_dir is not None:
            # Cache identification parameters:
            # - problem sets (patterns, validation_split, max_problems)
            # - question set (question_dir)
            # - dataset name (validation or train)
            # We only hash the parameters that cannot be easily represented by a string.
            hash_data = json.dumps({
                'patterns': problem_patterns,
                'validation_split': validation_split
            }).encode()
            hash_digest = hashlib.md5(hash_data).hexdigest()
            cache_path = os.path.join(cache_dir, str(max_problems), hash_digest, k)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            logging.info('Caching into: %s', cache_path)
            q = q.cache(cache_path)
        batches = questions.batch.batch(q, batch_size)
        if cache_mem:
            batches = batches.cache()
        if cache_dir is not None or cache_mem:
            questions.batch.preload(batches, k)
        return batches

    ques = {k: get_question_batches_dataset(p, k) for k, p in prob.items()}

    return prob, ques
