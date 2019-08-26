import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import (
    Input,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Concatenate,
    Dense,
    Dropout,
    Lambda,
    Multiply
)
from keras.initializers import RandomNormal, Constant, Ones
from keras.losses import mse
from normalization import ChannellNormalization
from scipy.stats import norm
from scipy.integrate import quad

EPSILON = 1E-9


# threshold subtraction layer
class ThresholdLayer(Layer):
    def __init__(self, filters, **kwargs):
        super(ThresholdLayer, self).__init__(**kwargs)
        self.filters = filters
        self.initializer = Constant(-self._calc_offset(self.filters))
        self.trainable = True

    def build(self, input_shape):
        self.threshold = self.add_weight(name='threshold',
                                         shape=(self.filters,),
                                         initializer=self.initializer,
                                         trainable=self.trainable)
        super(ThresholdLayer, self).build(input_shape)

    def call(self, x):
        return x + K.reshape(self.threshold, (1, 1, 1, self.filters))

    @staticmethod
    def _calc_offset(kernel_num):
        """calculate the initialized value of T,
        assuming the input is Gaussian distributed along channel axis with mean=0 and std=1 (~N(0, 1))
        """
        return quad(lambda x: kernel_num * x * norm.pdf(x, 0, 1) * (1 - norm.cdf(x, 0, 1)) ** (kernel_num - 1),
                    -np.float('Inf'), np.float('Inf'))[0]


# gain layer
class GainLayer(Layer):
    def __init__(self, **kwargs):
        super(GainLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gain = self.add_weight(name='gain',
                                    shape=(1,),
                                    initializer=Ones(),
                                    trainable=True)
        super(GainLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return self.gain * x


def model_builder(level, confidence=False, input_shape=(224, 224, 3)):
    if not confidence:
        if level == 1:
            return hierarchy1(input_shape)
        elif level == 2:
            return hierarchy2(input_shape)
        elif level == 3:
            return hierarchy3(input_shape)
    else:
        if level == 1:
            return hierarchy1_confidence(input_shape)
        elif level == 2:
            return hierarchy2_confidence(input_shape)
        elif level == 3:
            return hierarchy3_confidence(input_shape)


def _handle_dim_ordering():
    """Keras backend check
    """
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = -1
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _conv_unit(**params):
    """Helper to build a conventional convolution unit
    """
    filters = params["filters"]
    kernel_size = params.setdefault("kernel_size", (3, 3))
    strides = params.setdefault("strides", (1, 1))
    kernel_initializer = params.setdefault("kernel_initializer", RandomNormal(mean=0.0, stddev=0.1))
    use_bias = params.setdefault("use_bias", True)
    bias_initializer = params.setdefault("bias_initializer", "zeros")
    padding = params.setdefault("padding", "valid")
    pool_bool = params.setdefault("pool_bool", False)

    def f(inputs):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      use_bias=use_bias,
                      bias_initializer=bias_initializer)(inputs)

        bnorm = BatchNormalization(axis=-1, epsilon=EPSILON)(conv)
        relu = Activation("relu")(bnorm)
        if pool_bool:
            return MaxPooling2D(pool_size=(2, 2), padding='valid')(relu)
        else:
            return relu

    return f


def _reweight_unit(**params):
    """Helper to build a ReWU
    """
    kernel_size = params.setdefault("kernel_size", (1, 1))  # ReWU uses 1*1 convolution kernel
    strides = params.setdefault("strides", (1, 1))
    kernel_initializer = params.setdefault("kernel_initializer", RandomNormal(mean=0.0, stddev=0.1))
    padding = params.setdefault("padding", "valid")

    def f(inputs):
        input_batch_shape = K.int_shape(inputs)
        if 'filters' in params:
            filters = params["filters"]
        else:
            # default: the depth of the output of ReWU is equal to the input
            filters = input_batch_shape[CHANNEL_AXIS]
        conv = Conv2D(filters=filters, kernel_size=kernel_size, use_bias=False,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer)(inputs)

        chnorm = ChannellNormalization(axis=-1)(conv)
        threshold = ThresholdLayer(filters=filters)(chnorm)
        chmin = Lambda(lambda x: K.min(x, axis=-1, keepdims=True))(threshold)
        reweight_map = Activation("relu")(chmin)
        reweighted = Multiply()([inputs, reweight_map])
        return GainLayer()(reweighted)

    return f


def _concat_unit():
    """Helper to build a global average pooling and concatenation unit
    """
    def f(inputs):
        gap_vectors = list()
        for block in inputs:
            gap_vectors.append(GlobalAveragePooling2D()(block))
        return Concatenate()(gap_vectors)

    return f


def _fc_unit(**params):
    """Helper to build a fully-connected unit
    """
    units = params["units"]
    kernel_initializer = params.setdefault("kernel_initializer", RandomNormal(mean=0.0, stddev=0.02))
    bias_initializer = params.setdefault("bias_initializer", "zeros")
    relu_bool = params.setdefault("relu_bool", True)
    bnorm_bool = params.setdefault("bnorm_bool", False)
    dropout_bool = params.setdefault("dropout_bool", False)
    softmax_bool = params.setdefault("softmax_bool", False)
    sigmoid_bool = params.setdefault("sigmoid_bool", False)
    # the final activation layer can only be either softmax or sigmoid
    assert not (softmax_bool and sigmoid_bool)

    def f(inputs):
        fc = Dense(units=units,
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)(inputs)

        if bnorm_bool:
            bnorm = BatchNormalization(axis=-1, epsilon=EPSILON)(fc)
        else:
            bnorm = fc

        if relu_bool:
            relu = Activation("relu")(bnorm)
        else:
            relu = bnorm

        if dropout_bool:
            out = Dropout(rate=0.4)(relu)
        else:
            out = relu

        if softmax_bool:
            return Activation('softmax')(out)
        elif sigmoid_bool:
            return Activation('sigmoid')(out)
        else:
            return out

    return f


"""
Here are some custom layers that are used to build up models with confidence estimation branches.
Building up a multi-input/output model in Keras is not as neat as building up a single input/output model.
We treat the following metrics as the outputs of the model: 
    1. angular error
    2. mean squared error
    3. task error
    4. regularization error
During training, (task error + lambda * regularization error) is the final loss to be minimized,
by setting the loss weights to [1, lambda, 0, 0] for these four outputs.
Angular error and mean squared error are for performance evaluation only.
"""


def angular_error_layer(args):
    y_true, y_pred = args
    p = K.sum(K.l2_normalize(y_true, axis=-1) * K.l2_normalize(y_pred, axis=-1), axis=-1)
    p = K.clip(p, EPSILON, 1. - EPSILON)
    return 180 * tf.acos(p) / np.pi


def mean_squared_error_layer(args):
    y_true, y_pred = args
    return mse(y_true, y_pred)


def task_error_layer(args):
    y_true, y_pred, conf = args
    predictions_adjusted = (conf * y_pred) + ((1 - conf) * y_true)
    # task_error_part1 is the basic task error introduced in the paper
    task_error_part1 = mse(y_true, predictions_adjusted)
    # this mse threshold parameter should be determined by evaluating the naive network
    # and find a appropriate value that corresponds to the max allowable angular error
    task_err_threshold = K.variable(0.0006)
    # task_error_part2 imposes stronger penalties to those sample with task error larger than task_err_threshold
    task_error_part2 = 10 * K.relu(task_error_part1 - task_err_threshold)
    return task_error_part1 + task_error_part2


def regularization_error_layer(conf):
    return -K.log(K.clip(conf, EPSILON, 1. - EPSILON))


# TODO: use an abstracted function to simplify the network architecture constructions
def hierarchy1(input_shape):
    """Builds a custom Hierarchy-1 network.
    Args:
        input_shape: The input_image shape in the form (nb_rows, nb_cols, nb_channels)
    Returns:
        The Keras `Model`.
    """
    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")
    # Permute dimension order if necessary
    if K.image_dim_ordering() == 'th':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])

    input_image = Input(shape=input_shape)
    conv_unit1 = _conv_unit(filters=32, strides=(2, 2), use_bias=False)(input_image)

    input_norm = BatchNormalization(axis=-1, epsilon=EPSILON)(input_image)
    rewu0 = _reweight_unit(filters=16, kernel_initializer=RandomNormal(mean=0.0, stddev=0.06))(input_norm)
    rewu1 = _reweight_unit(kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(conv_unit1)

    concat = _concat_unit()([rewu0, rewu1])

    fc1 = _fc_unit(units=64, bnorm_bool=True, dropout_bool=True,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(concat)
    fc2 = _fc_unit(units=32, bnorm_bool=True, dropout_bool=True,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(fc1)
    fc3 = _fc_unit(units=16, bnorm_bool=True, dropout_bool=True,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.03))(fc2)
    estimate = _fc_unit(units=3, softmax_bool=True,
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.06))(fc3)
    model = Model([input_image], [estimate])

    return model


def hierarchy1_confidence(input_shape):
    """Builds a custom Hierarchy-1 network with confidence estimation branch.
    Args:
        input_shape: The input_image shape in the form (nb_rows, nb_cols, nb_channels)
    Returns:
        The Keras `Model`.
    """
    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")
    # Permute dimension order if necessary
    if K.image_dim_ordering() == 'th':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])

    groundtruth = Input(shape=(3,))

    input_image = Input(shape=input_shape)
    conv_unit1 = _conv_unit(filters=32, strides=(2, 2), use_bias=False)(input_image)

    input_norm = BatchNormalization(axis=-1, epsilon=EPSILON)(input_image)
    rewu0 = _reweight_unit(filters=16)(input_norm)
    rewu1 = _reweight_unit()(conv_unit1)

    concat = _concat_unit()([rewu0, rewu1])

    fc1 = _fc_unit(units=64, bnorm_bool=True, dropout_bool=True)(concat)
    fc2 = _fc_unit(units=32, bnorm_bool=True, dropout_bool=True)(fc1)
    fc3 = _fc_unit(units=16, bnorm_bool=True, dropout_bool=True)(fc2)
    estimate = _fc_unit(units=3, softmax_bool=True)(fc3)

    # confidence estimation branch
    fc1_conf = _fc_unit(units=64, bnorm_bool=True, dropout_bool=True)(concat)
    fc2_conf = _fc_unit(units=32, bnorm_bool=True, dropout_bool=True)(fc1_conf)
    fc3_conf = _fc_unit(units=16, bnorm_bool=True, dropout_bool=True, relu_bool=False)(fc2_conf)
    confidence = _fc_unit(units=1, relu_bool=False, sigmoid_bool=True)(fc3_conf)

    mean_squared_error = Lambda(mean_squared_error_layer, output_shape=(1,),
                                name='mean_squared_error')([groundtruth, estimate])
    ang_err = Lambda(angular_error_layer, output_shape=(1,),
                     name='ang_error')([groundtruth, estimate])
    task_err = Lambda(task_error_layer, output_shape=(1,),
                      name='task_error')([groundtruth, estimate, confidence])
    regularization_err = Lambda(regularization_error_layer, output_shape=(1,),
                                name='regularization_error')(confidence)
    model = Model(inputs=[input_image, groundtruth],
                  outputs=[task_err, regularization_err, ang_err, mean_squared_error, estimate, confidence])

    return model


def hierarchy2(input_shape):
    """Builds a custom Hierarchy-2 network.
    Args:
        input_shape: The input_image shape in the form (nb_rows, nb_cols, nb_channels)
    Returns:
        The Keras `Model`.
    """
    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")
    # Permute dimension order if necessary
    if K.image_dim_ordering() == 'th':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])

    input_image = Input(shape=input_shape)
    conv_unit1 = _conv_unit(filters=32, strides=(2, 2), use_bias=False)(input_image)
    conv_unit2 = _conv_unit(filters=32, use_bias=False)(conv_unit1)

    input_norm = BatchNormalization(axis=-1, epsilon=EPSILON)(input_image)
    rewu0 = _reweight_unit(filters=16, kernel_initializer=RandomNormal(mean=0.0, stddev=0.06))(input_norm)
    rewu1 = _reweight_unit(kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(conv_unit1)
    rewu2 = _reweight_unit(kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(conv_unit2)

    concat = _concat_unit()([rewu0, rewu1, rewu2])

    fc1 = _fc_unit(units=128, bnorm_bool=True, dropout_bool=True,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(concat)
    fc2 = _fc_unit(units=64, bnorm_bool=True, dropout_bool=True,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(fc1)
    fc3 = _fc_unit(units=32, bnorm_bool=True, dropout_bool=True,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(fc2)
    estimate = _fc_unit(units=3, softmax_bool=True,
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.03))(fc3)
    model = Model([input_image], [estimate])

    return model


def hierarchy2_confidence(input_shape):
    """Builds a custom Hierarchy-3 network with confidence estimation branch.
    Args:
        input_shape: The input_image shape in the form (nb_rows, nb_cols, nb_channels)
    Returns:
        The Keras `Model`.
    """
    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")
    # Permute dimension order if necessary
    if K.image_dim_ordering() == 'th':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])

    groundtruth = Input(shape=(3,))

    input_image = Input(shape=input_shape)
    conv_unit1 = _conv_unit(filters=32, strides=(2, 2), use_bias=False)(input_image)
    conv_unit2 = _conv_unit(filters=32, use_bias=False)(conv_unit1)

    input_norm = BatchNormalization(axis=-1, epsilon=EPSILON)(input_image)
    rewu0 = _reweight_unit(filters=16)(input_norm)
    rewu1 = _reweight_unit()(conv_unit1)
    rewu2 = _reweight_unit()(conv_unit2)

    concat = _concat_unit()([rewu0, rewu1, rewu2])

    fc1 = _fc_unit(units=128, bnorm_bool=True, dropout_bool=True)(concat)
    fc2 = _fc_unit(units=64, bnorm_bool=True, dropout_bool=True)(fc1)
    fc3 = _fc_unit(units=32, bnorm_bool=True, dropout_bool=True)(fc2)
    estimate = _fc_unit(units=3, softmax_bool=True)(fc3)

    # confidence estimation branch
    fc1_conf = _fc_unit(units=128, bnorm_bool=True, dropout_bool=True)(concat)
    fc2_conf = _fc_unit(units=64, bnorm_bool=True, dropout_bool=True)(fc1_conf)
    fc3_conf = _fc_unit(units=32, bnorm_bool=True, dropout_bool=True, relu_bool=False)(fc2_conf)
    confidence = _fc_unit(units=1, relu_bool=False, sigmoid_bool=True)(fc3_conf)

    mean_squared_error = Lambda(mean_squared_error_layer, output_shape=(1,),
                                name='mean_squared_error')([groundtruth, estimate])
    ang_err = Lambda(angular_error_layer, output_shape=(1,),
                     name='ang_error')([groundtruth, estimate])
    task_err = Lambda(task_error_layer, output_shape=(1,),
                      name='task_error')([groundtruth, estimate, confidence])
    regularization_err = Lambda(regularization_error_layer, output_shape=(1,),
                                name='regularization_error')(confidence)
    model = Model(inputs=[input_image, groundtruth],
                  outputs=[task_err, regularization_err, ang_err, mean_squared_error, estimate, confidence])

    return model


def hierarchy3(input_shape):
    """Builds a custom Hierarchy-3 network.
    Args:
        input_shape: The input_image shape in the form (nb_rows, nb_cols, nb_channels)
    Returns:
        The Keras `Model`.
    """
    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")
    # Permute dimension order if necessary
    if K.image_dim_ordering() == 'th':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])

    input_image = Input(shape=input_shape)
    conv_unit1 = _conv_unit(filters=32, strides=(2, 2), use_bias=False)(input_image)
    conv_unit2 = _conv_unit(filters=32, use_bias=False)(conv_unit1)
    conv_unit3 = _conv_unit(filters=64, pool_bool=True, use_bias=False)(conv_unit2)

    input_norm = BatchNormalization(axis=-1, epsilon=EPSILON)(input_image)
    rewu0 = _reweight_unit(filters=16, kernel_initializer=RandomNormal(mean=0.0, stddev=0.06))(input_norm)
    rewu1 = _reweight_unit(kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(conv_unit1)
    rewu2 = _reweight_unit(kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(conv_unit2)
    rewu3 = _reweight_unit(kernel_initializer=RandomNormal(mean=0.0, stddev=0.03))(conv_unit3)

    concat = _concat_unit()([rewu0, rewu1, rewu2, rewu3])

    fc1 = _fc_unit(units=256, bnorm_bool=True, dropout_bool=True,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(concat)
    fc2 = _fc_unit(units=128, bnorm_bool=True, dropout_bool=True,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(fc1)
    fc3 = _fc_unit(units=64, bnorm_bool=True, dropout_bool=True,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(fc2)
    estimate = _fc_unit(units=3, softmax_bool=True,
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.04))(fc3)
    model = Model([input_image], [estimate])

    return model


def hierarchy3_confidence(input_shape):
    """Builds a custom Hierarchy-3 network with confidence estimation branch.
    Args:
        input_shape: The input_image shape in the form (nb_rows, nb_cols, nb_channels)
    Returns:
        The Keras `Model`.
    """
    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")
    # Permute dimension order if necessary
    if K.image_data_format() == 'th':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])

    groundtruth = Input(shape=(3,))

    input_image = Input(shape=input_shape)
    conv_unit1 = _conv_unit(filters=32, strides=(2, 2), use_bias=False)(input_image)
    conv_unit2 = _conv_unit(filters=32, use_bias=False)(conv_unit1)
    conv_unit3 = _conv_unit(filters=64, pool_bool=True, use_bias=False)(conv_unit2)

    input_norm = BatchNormalization(axis=-1, epsilon=EPSILON)(input_image)
    rewu0 = _reweight_unit(filters=16)(input_norm)
    rewu1 = _reweight_unit()(conv_unit1)
    rewu2 = _reweight_unit()(conv_unit2)
    rewu3 = _reweight_unit()(conv_unit3)

    concat = _concat_unit()([rewu0, rewu1, rewu2, rewu3])

    fc1 = _fc_unit(units=256, bnorm_bool=True, dropout_bool=True)(concat)
    fc2 = _fc_unit(units=128, bnorm_bool=True, dropout_bool=True)(fc1)
    fc3 = _fc_unit(units=64, bnorm_bool=True, dropout_bool=True)(fc2)
    estimate = _fc_unit(units=3, softmax_bool=True)(fc3)

    # confidence estimation branch
    fc1_conf = _fc_unit(units=256, bnorm_bool=True, dropout_bool=True)(concat)
    fc2_conf = _fc_unit(units=128, bnorm_bool=True, dropout_bool=True)(fc1_conf)
    fc3_conf = _fc_unit(units=64, bnorm_bool=True, dropout_bool=True, relu_bool=False)(fc2_conf)
    confidence = _fc_unit(units=1, relu_bool=False, sigmoid_bool=True)(fc3_conf)

    mean_squared_error = Lambda(mean_squared_error_layer, output_shape=(1,),
                                name='mean_squared_error')([groundtruth, estimate])
    ang_err = Lambda(angular_error_layer, output_shape=(1,),
                     name='ang_error')([groundtruth, estimate])
    task_err = Lambda(task_error_layer, output_shape=(1,),
                      name='task_error')([groundtruth, estimate, confidence])
    regularization_err = Lambda(regularization_error_layer, output_shape=(1,),
                                name='regularization_error')(confidence)
    model = Model(inputs=[input_image, groundtruth],
                  outputs=[task_err, regularization_err, ang_err, mean_squared_error, estimate, confidence])

    return model
