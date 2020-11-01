import logging
import os

from tqdm import tqdm

from proving import load_questions


def get_datasets_batched(question_dir, problems, batch_size, cache_dir=None):
    logging.info('Batch size: %d', batch_size)
    questions = get_datasets(question_dir, problems, cache_dir)
    res = {}
    for k, q in questions.items():
        batches = load_questions.questions_batched_dataset(q, batch_size)
        res[k] = batches
    return res


def get_datasets(question_dir, problems, cache_dir=None):
    res = {}
    for k, p in problems.items():
        questions = load_questions.get_dataset(question_dir, p)
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, k)
            logging.info('Caching into: %s', cache_path)
            # Parameters: problems, question set (path), dataset (train or validation), batch size
            questions = questions.cache(cache_path)
        res[k] = questions
    return res


def preload(dataset):
    n_batches = 0
    n_elements = 0
    for batch in tqdm(dataset, unit='batch', desc='Preloading batches'):
        n_batches += 1
        n_elements += len(batch['problems'])
    logging.info(f'Number of batches: {n_batches}')
    logging.info(f'Number of problems with questions: {n_elements}')
