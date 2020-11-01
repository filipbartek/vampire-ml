import logging

import more_itertools
import tensorflow as tf
from tqdm import tqdm

from . import individual


def get_datasets(question_dir, problems, batch_size, cache_dir=None):
    logging.info('Batch size: %d', batch_size)
    questions = individual.get_datasets(question_dir, problems, cache_dir)
    res = {}
    for k, q in questions.items():
        batches = batch(q, batch_size)
        res[k] = batches
    return res


def batch(dataset, batch_size, row_splits_dtype=tf.int64):
    def gen():
        for b in more_itertools.chunked(dataset, batch_size):
            problems_tensor = tf.stack([e['problem'] for e in b])
            questions_tensor = ragged_from_tensors((e['questions'] for e in b), row_splits_dtype)
            yield {'problems': problems_tensor, 'questions': {'flat_values': questions_tensor.flat_values,
                                                              'nested_row_splits': questions_tensor.nested_row_splits}}

    output_types = {
        'problems': dataset.element_spec['problem'].dtype,
        'questions': {
            'flat_values': dataset.element_spec['questions'].dtype,
            'nested_row_splits': (row_splits_dtype, row_splits_dtype)
        }
    }
    output_shapes = {
        # Note: We specify None instead of `batch_size` because the last batch may be smaller than `batch_size`.
        'problems': tf.TensorShape([None]),
        'questions': {
            'flat_values': tf.TensorShape([None]),
            'nested_row_splits': (tf.TensorShape([None]), tf.TensorShape([None]))
        }
    }
    return tf.data.Dataset.from_generator(gen, output_types, output_shapes)


def ragged_from_tensors(tensors, row_splits_dtype):
    flat_values, nested_row_lengths = matrices_to_ragged_precursors(tensors, row_splits_dtype)
    return tf.RaggedTensor.from_nested_row_lengths(flat_values, nested_row_lengths)


def matrices_to_ragged_precursors(tensors, row_lengths_dtype=tf.int64):
    tensors = list(tensors)
    assert all(len(t.shape) == 2 for t in tensors)
    flat_tensors = [tf.reshape(t, (-1,)) for t in tensors]
    flat_values = tf.concat(flat_tensors, 0)
    shapes = tf.transpose(tf.stack([t.shape for t in tensors]))
    # The default dtype of nested row splits is tf.int64. We follow the convention to avoid collisions.
    shapes = tf.cast(shapes, row_lengths_dtype)
    nested_row_lengths = (
        shapes[0],
        tf.repeat(shapes[1], shapes[0])
    )
    return flat_values, nested_row_lengths


def preload(dataset):
    n_batches = 0
    n_elements = 0
    for batch in tqdm(dataset, unit='batch', desc='Preloading batches'):
        n_batches += 1
        n_elements += len(batch['problems'])
    logging.info(f'Number of batches: {n_batches}')
    logging.info(f'Number of problems with questions: {n_elements}')
