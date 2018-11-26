import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import mocha.utils as utils
from mocha.logger import Logger
from mocha.story import Story


class Model:
    class Layers:
        def __init__(self):
            self.value = {}

        def add(self, name, value):
            self.value[name] = value

    class Weights:
        def __init__(self):
            self.value = {}

        def add(self, name, value):
            self.value[name] = value

    class Biases:
        def __init__(self):
            self.value = {}

        def add(self, name, value):
            self.value[name] = value

    class HyperParameters:
        def __init__(self, values={}):
            self.log = Logger("hyper_parameters")
            self.learning_rate = 0.001
            self.batch_size = 50
            self.n_epochs = 5
            self.update(values)

        def update(self, values):
            for key, value in values.items():
                if key == "learning_rate":
                    self.learning_rate = value
                else:
                    if key == "batch_size":
                        self.batch_size = value
                    else:
                        if key == "n_epochs":
                            self.n_epochs = value
                        else:
                            self.log.write("Ignored hyper parameter: {}".format(key), indent=1,
                                           level=self.log.Level.WARNING)

    class InputParameters:
        def __init__(self, values={}):
            self.log = Logger("input_parameters")
            self.width = 28
            self.height = 28
            self.channels = 1
            self.update(values)

        def update(self, values):
            for key, value in values.items():
                if key == "width":
                    self.width = value
                else:
                    if key == "height":
                        self.height = value
                    else:
                        if key == "channels":
                            self.channels = value
                        else:
                            self.log.write("Ignored input parameter: {}".format(key), indent=1,
                                           level=self.log.Level.WARNING)

    class Parameters:
        """
        Default parameters of the model
        filter_size: height and width of the filters
        window_size: height and width of the sliding windows
        stride: step between each window
        padding: border handling
        """
        def __init__(self, values={}):
            self.log = Logger("parameters")
            self.filter_size = 5
            self.window_size = 2
            self.stride = 1
            self.padding = 'SAME'
            self.update(values)

        def update(self, values):
            for key, value in values.items():
                if key == "filter_size":
                    self.filter_size = value
                else:
                    if key == "window_size":
                        self.window_size = value
                    else:
                        if key == "stride":
                            self.stride = value
                        else:
                            if key == "padding":
                                self.padding = value
                            else:
                                self.log.write("Ignored default parameter: {}".format(key), indent=1,
                                               level=self.log.Level.WARNING)

    def __init__(self, name, output_layer_size, loss_function, optimizer, parameters={}, input_parameters={}, hyper_parameters={}):
        # the story of the model
        self.story = Story("model")
        self.log = Logger("model")
        self.name = name

        tf.logging.set_verbosity(tf.logging.ERROR)

        self.output_layer_size = output_layer_size

        # neural network
        self.layers = self.Layers()
        self.weights = self.Weights()
        self.biases = self.Biases()

        # parameters
        self.parameters = self.Parameters(parameters)
        self.input_parameters = self.InputParameters(input_parameters)
        self.hyper_parameters = self.HyperParameters(hyper_parameters)

        # loss and optimizer
        self.loss_function_name = loss_function
        self.optimizer_name = optimizer

        # parameters defined on the flight
        self.input_layer_size = None
        self.x = None
        self.y = None
        self.session = None
        self.features = None
        self.labels = None
        self.stories = []

        self.log.write("model has been initialized")

    def display_feature(self, index):
        plt.imshow(self.features[index].reshape((self.input_parameters.width, self.input_parameters.height)))
        plt.show()

    def init(self):
        self.input_layer_size = self.input_parameters.width * self.input_parameters.height \
                                * self.input_parameters.channels

        self.x = tf.placeholder(tf.float32, [None, self.input_layer_size], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.output_layer_size], name='y')

        # init a reshape layer to fit the input to the first layer
        self.layers.add("reshape", tf.reshape(self.x, [-1, self.input_parameters.width,
                                                       self.input_parameters.height, 1]))

    def open_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def last_layer(self):
        last_key = list(self.layers.value.keys())[-1]
        return self.layers.value[last_key]

    def add_conv2d(self, name):
        self.layers.add(name,
                        tf.nn.relu(tf.nn.conv2d(self.last_layer(), self.weights.value[name],
                                                strides=[1, self.parameters.stride, self.parameters.stride, 1],
                                                padding=self.parameters.padding,
                                                name=name) + self.biases.value[name]))

    def add_maxpool(self, name):
        self.layers.add(name,
                        tf.nn.max_pool(self.last_layer(),
                                       ksize=[1, self.parameters.window_size, self.parameters.window_size, 1],
                                       strides=[1, self.parameters.window_size, self.parameters.window_size, 1],
                                       padding=self.parameters.padding))

    def add_flatten(self, name):
        self.layers.add(name,
                        tf.reshape(self.last_layer(),
                                   [-1, self.last_layer().shape[1] * self.last_layer().shape[2]
                                    * self.last_layer().shape[3]]))

    def add_fully_connected(self, name, activation=tf.identity):
        # X * W + b
        self.layers.add(name,
                        activation(tf.matmul(self.last_layer(), self.weights.value[name]) + self.biases.value[name]))

    def __init_loss_function(self):
        """
        Initialize the loss function just before running a session
        """
        if self.loss_function_name == "softmax_cross_entropy_with_logits":
            self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.last_layer(), labels=self.y))

    def __init_optimizer(self):
        """
        Initialize the optimizer just after the loss function has been initialized
        """
        if self.optimizer_name == "adam":
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_parameters.learning_rate).minimize(self.loss_function)

    def __init_accuracy(self):
        # compute accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.last_layer(), 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))

    def get_slice_limits(self, i):
        start = i * self.hyper_parameters.batch_size
        end = start + self.hyper_parameters.batch_size

        return start, end

    def train_init_data(self, features, labels):
        self.features = features
        self.labels = labels

    def train_init_functions(self):
        self.__init_loss_function()
        self.__init_optimizer()
        self.__init_accuracy()

    def train(self, features=None, labels=None):
        """
        Train the model
        :param features: the features on which to train the model
        :param labels: the target
        :return: True if succeed
        """
        if features:
            self.features = features

        if labels:
            self.labels = labels

        self.train_init_functions()

        self.open_session()

        self.log.write("training start")
        self.story.new_session("{}_train".format(self.name))

        for epoch in range(self.hyper_parameters.n_epochs):
            # create a story for the current epoch
            self.story.new_epoch(epoch)
            self.log.write("Epoch %02d" % (epoch + 1), indent=1)

            # batch steps calculation
            n_steps = len(self.labels) // self.hyper_parameters.batch_size

            for step in range(n_steps):

                # get batches
                start, end = self.get_slice_limits(step)
                batch_xs, batch_ys = self.features[start:end], self.labels[start:end]

                # calculation
                optimizer_res, loss_res, accuracy_res = self.session.run([
                    self.optimizer, self.loss_function, self.accuracy],
                    feed_dict={self.x: batch_xs, self.y: batch_ys})

                if step % 100 == 0:
                    self.log.write("accuracy = {}, loss={}".format(accuracy_res, loss_res).format(self.compute_accuracy()), indent=2)
                    self.story.new_acc(accuracy_res)
                    self.story.new_loss(loss_res)

            # end of the epoch auto-compute accuracy and loss
            self.story.close_epoch()
            self.log.write("train accuracy: {}".format(self.story.epoch()["acc_mean"]), indent=1, level=self.log.Level.SUCCESS)

        self.story.close_session()
        self.log.write("training finished", level=self.log.Level.SUCCESS)

        return True

    def test(self, features, labels):
        self.story.new_session("{}_test".format(self.name))
        self.story.session()["acc"] = self.session.run(self.accuracy, feed_dict={self.x:features, self.y:labels})
        self.log.write("test accuracy: {}".format(self.story.session()["acc"]), indent=1,
                       level=self.log.Level.SUCCESS)
        self.story.close_session()
        self.log.write("test finished", level=self.log.Level.SUCCESS)

    def compute_accuracy(self):
        """
        Calculate the accuracy using a batch mode to prevent out of memory errors
        :return: accuracy
        """
        accuracies = []
        for start, end, ratio in utils.samples_indexes(self.features):
            accuracies.append(self.session.run(
                self.accuracy,
                feed_dict={self.x: self.features[start:end], self.y: self.labels[start:end]}))

        res = np.mean(accuracies)
        return res
