import logging
import os

from . import questions


def get_datasets(prob, questions_dir, batch_size, cache_dir=None, cache_mem=False):
    def get_question_batches_dataset(p, k):
        q = questions.individual.get_dataset(questions_dir, p)
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, k)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            logging.info('Caching into: %s', cache_path)
            q = q.cache(cache_path)
        batches = questions.batch.batch(q, batch_size)
        if cache_mem:
            batches = batches.cache()
        if cache_dir is not None or cache_mem:
            questions.batch.preload(batches, k)
        return batches

    return {k: get_question_batches_dataset(p, k) for k, p in prob.items()}
