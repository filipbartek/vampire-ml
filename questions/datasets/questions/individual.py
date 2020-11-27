import functools
import itertools
import os

import tensorflow as tf

from proving.utils import py_str


def dict_to_dataset(questions, problems, normalize=True, dtype=tf.float32):
    def gen():
        for problem in problems:
            try:
                q = tf.constant(questions[py_str(problem)], dtype=dtype)
                # Let n be the number of symbols in the problem.
                n = tf.shape(q)[1]
                n = tf.cast(n, q.dtype)
                tf.debugging.assert_greater_equal(n, tf.reduce_max(q))
                tf.debugging.assert_less_equal(-n, tf.reduce_min(q))
                if normalize:
                    # We scale the question matrix by a factor that makes questions from problems of various sizes commensurable.
                    # We set the factor so that if symbol cost is 1 for all symbols, then precedence cost is 1 for all precedences.
                    q *= 2 / (n * (n + 1))
                    tf.debugging.assert_greater_equal(2 / (n + 1), tf.reduce_max(q))
                    tf.debugging.assert_less_equal(-2 / (n + 1), tf.reduce_min(q))
                yield {'problem': problem, 'questions': q}
            except KeyError:
                pass

    output_types = {'problem': tf.string, 'questions': dtype}
    output_shapes = {'problem': tf.TensorShape([]), 'questions': tf.TensorShape([None, None])}

    # Note: We cannot construct the dataset using `from_tensor_slices` because that function requires all 'questions'
    # tensors to have the same shape.
    return tf.data.Dataset.from_generator(gen, output_types, output_shapes)


def get_dataset(question_dir, problems, dtype=tf.float32):
    def gen():
        for problem in problems:
            pattern = os.path.join(question_dir, f'{py_str(problem)}_*.q')
            filenames = tf.io.gfile.glob(pattern)
            questions_iterator = map(functools.partial(filename_to_question, dtype=dtype), filenames)
            questions_list = list(questions_iterator)
            if len(questions_list) >= 1:
                yield {'problem': problem, 'questions': tf.stack(questions_list)}

    output_types = {'problem': tf.string, 'questions': dtype}
    output_shapes = {'problem': tf.TensorShape([]), 'questions': tf.TensorShape([None, None])}

    # Note: We cannot construct the dataset using `from_tensor_slices` because that function requires all 'questions'
    # tensors to have the same shape.
    return tf.data.Dataset.from_generator(gen, output_types, output_shapes)


def filename_to_question(filename, dtype):
    it = iter(tf.data.TextLineDataset(filename))
    precedences = tuple(map(functools.partial(line_to_precedence, dtype=dtype), itertools.islice(it, 2)))
    # TODO: Invert precedences.
    precedence_diff = precedences[0] - precedences[1]
    polarity = next(it)
    # res = tf.cond(polarity == '>', lambda: precedence_diff * -1, lambda: precedence_diff)
    if polarity == '>':
        res = precedence_diff * -1
    else:
        res = precedence_diff
    # Normalize
    n = res.shape[0]
    res = res * 2 / (n * (n + 1))
    return res


def line_to_precedence(x, dtype=tf.int32, sep=','):
    """Convert a string with comma-separated numbers to a tensor.

    Similar to `numpy.fromstring`.
    """
    return tf.strings.to_number(tf.strings.split(x, sep=sep), out_type=dtype)
