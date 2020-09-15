import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm


def evaluate_weights(weights, data_test, data_train, loss_fn):
    k = len(weights)
    model = get_model(k, weights=[weights.reshape(-1, 1)])
    return evaluate(model, data_test, data_train, loss_fn, extract_weights=True)


def get_model(k, weights=None, use_bias=False, hidden_units=0):
    # Row: problem -> symbol
    symbol_embeddings = keras.Input(shape=k, name='symbol_embeddings')
    x = symbol_embeddings
    if hidden_units > 0:
        x = layers.Dense(hidden_units, 'relu')(x)
    symbol_costs_layer = layers.Dense(1, use_bias=use_bias, name='symbol_costs')
    symbol_costs = symbol_costs_layer(x)
    if weights is not None:
        symbol_costs_layer.set_weights(weights)
    # Row: problem -> question -> symbol
    ranking_difference = keras.Input(shape=1, name='ranking_difference')
    question_symbols = keras.Input(shape=1, name='question_symbols', dtype=tf.int32)
    symbol_costs_tiled = layers.Flatten(name='symbol_costs_tiled')(tf.gather(symbol_costs, question_symbols))
    potentials = layers.multiply([symbol_costs_tiled, ranking_difference])
    segment_ids = keras.Input(shape=1, name='segment_ids', dtype=tf.int32)
    precedence_pair_logit = tf.math.segment_sum(potentials, keras.backend.flatten(segment_ids))
    precedence_pair_logit = layers.Flatten(name='logits')(precedence_pair_logit)
    return keras.Model(inputs=[symbol_embeddings, ranking_difference, question_symbols, segment_ids],
                       outputs=precedence_pair_logit)


def evaluate(model, data_test, data_train, loss_fn, test_summary_writer=None, train_summary_writer=None,
             extract_weights=False):
    record = {}
    if extract_weights:
        weights = model.get_layer('symbol_costs').get_weights()
        for weight_i, weight in enumerate(weights):
            record[('weight', weight_i)] = np.squeeze(weight)
            record[('weight_normalized', weight_i)] = np.squeeze(sklearn.preprocessing.normalize(weight, axis=0))
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.text('weights', str(weights))
    for name, value in test_step(model, data_test, loss_fn).items():
        record[('test', name)] = value.numpy()
        if test_summary_writer is not None:
            with test_summary_writer.as_default():
                tf.summary.scalar(name, value)
    for name, value in test_step(model, data_train, loss_fn).items():
        record[('train', name)] = value.numpy()
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar(name, value)
    return record


def test_step(model, data, loss_fn):
    if data is None:
        return {}
    xs = data['xs']
    sample_weight = data['sample_weight']
    xs = ({'ranking_difference': x['ranking_difference'], 'question_symbols': x['question_symbols'],
           'segment_ids': x['segment_ids'], 'symbol_embeddings': x['symbol_embeddings_predicate']} for x in xs)
    logits = tf.concat([model(x, training=False) for x in tqdm(xs, unit='batch', desc='Evaluating on batches', disable=len(data['xs']) <= 1)], axis=0)
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('probs', tf.sigmoid(logits))
    assert len(logits) == len(sample_weight)
    res = {'loss': loss_fn(np.ones((len(sample_weight), 1), dtype=np.bool), logits, sample_weight=sample_weight) / np.mean(sample_weight)}
    metrics = {
        'accuracy': keras.metrics.BinaryAccuracy(threshold=0),
        'crossentropy': keras.metrics.BinaryCrossentropy(from_logits=True)
    }
    for name, metric in metrics.items():
        metric.update_state(np.ones((len(sample_weight), 1), dtype=np.bool), logits, sample_weight=sample_weight)
        res[name] = metric.result()
    return res
