import tensorflow as tf
import bson


def object_id():
    return bson.objectid.ObjectId()


def samples_indexes(collection, sample_size=20000, truncate=True):
    """
    return the sample indexes for the given data
    :param collection: data
    :param sample_size: size of each sample
    :param truncate: get only the samples of the same size
    :return:
    """
    collection_size = len(collection)

    # calculate sample count
    sample_count = int(collection_size / sample_size) + int(collection_size % sample_size != 0)

    # initialize sample max index
    samples_indexes = [(i*sample_size, i*sample_size + sample_size,1.) for i in range(sample_count)]

    # limit of the last batch
    if collection_size % sample_size != 0:
        if truncate:
            samples_indexes.remove(samples_indexes[-1])
        else:
            last_sample_max_index = samples_indexes[-1][0] + collection_size % sample_size

            samples_indexes[-1] = (samples_indexes[-1][0], last_sample_max_index, (last_sample_max_index - samples_indexes[-1][0]) / sample_size)

    return samples_indexes


def weight_variable(shape):
    """
    Initialize the weight Variable with values from a normal distribution
    :param shape: the shape of the weight object
    :return: tf.Variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1, seed=10)
    return tf.Variable(initial, name='weight')


def bias_variable(shape, value=0.1):
    """
    Initialize the bias Variable with constant values
    :param shape: the shape of the bias object
    :param value: values of the constant
    :return: tf.Variable
    """
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name='bias')

