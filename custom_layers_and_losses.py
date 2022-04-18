import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K
import math

########################  CAPAS PERSONALIZADAS ########################################################

# Capa ArcFace
# Implementacion basada en:
# https://github.com/4uiiurz1/keras-arcface/blob/master/metrics.py
# https://github.com/peteryuX/arcface-tf2/blob/master/modules/layers.py
class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizer

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_classes": self.n_classes,
            "logits_scale": self.s,
            "margin": self.m,
            "regularizer": self.regularizer
        })
        return config

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, x, y):
        # one-hot encode the targets
        y = tf.one_hot(tf.cast(y, tf.int32), depth=self.n_classes,
                       name='one_hot_mask')

        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


# Codigo adaptado de TensorFlow 1 a TensorFlow 2 a partir de:
# https://github.com/DHZS/tf-deformable-conv-layer
class DeformableConvLayer(Conv2D):
    """Only support "channel last" data format"""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_deformable_group=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.
        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=filters.
        """
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.kernel = None
        self.bias = None
        self.offset_layer_kernel = None
        self.offset_layer_bias = None
        if num_deformable_group is None:
            num_deformable_group = filters
        if filters % num_deformable_group != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # kernel_shape = self.kernel_size + (input_dim, self.filters)
        # we want to use depth-wise conv
        kernel_shape = self.kernel_size + (self.filters * input_dim, 1)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.num_deformable_group
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            shape=self.kernel_size + (input_dim, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.offset_layer_bias = self.add_weight(
            name='offset_layer_bias',
            shape=(offset_num * 2,),
            initializer=tf.zeros_initializer(),
            # initializer=tf.random_uniform_initializer(-5, 5),
            regularizer=self.bias_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        # get offset, shape [batch_size, out_h, out_w, filter_h, * filter_w * channel_out * 2]
        offset = tf.nn.conv2d(inputs,
                              filters=self.offset_layer_kernel,
                              strides=[1, *self.strides, 1],
                              padding=self.padding.upper(),
                              dilations=[1, *self.dilation_rate, 1])
        offset += self.offset_layer_bias

        # add padding if needed
        inputs = self._pad_input(inputs)

        # some length
        # since undefined batch sizes are handled differently in tf1 and tf2, (as placeholders and
        # 'None' shapes respectively), shape has to be obtained by the dynamic tf.shape.
        # batch_size = int(inputs.get_shape()[0])
        batch_size = tf.shape(inputs)[0]
        channel_in = int(inputs.get_shape()[-1])
        in_h, in_w = [int(i) for i in inputs.get_shape()[1: 3]]  # input feature map size
        out_h, out_w = [int(i) for i in offset.get_shape()[1: 3]]  # output feature map size
        filter_h, filter_w = self.kernel_size

        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, -1, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]

        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [batch_size, 1, 1, 1, self.num_deformable_group]) for i in [y, x]]
        # y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
        y, x = [tf.reshape(i, [tf.shape(i)[0], *i.shape[1:3], -1]) for i in [y, x]]
        # y = tf.reshape(y, [*tf.shape(y)[0:3],-1])
        # x = tf.reshape(x, [*tf.shape(x)[0:3],-1])
        # y, x = [tf.to_float(i) for i in [y, x]]
        y, x = [tf.cast(i, tf.float32) for i in [y, x]]

        # add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0, in_h - 1)
        x = tf.clip_by_value(x, 0, in_w - 1)

        # get four coordinates of points around (x, y)
        # y0, x0 = [tf.to_int32(tf.floor(i)) for i in [y, x]]
        y0, x0 = [tf.cast(tf.floor(i), tf.int32) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [DeformableConvLayer._get_pixel_values_at_point(inputs, i) for i in indices]

        # cast to float
        # x0, x1, y0, y1 = [tf.to_float(i) for i in [x0, x1, y0, y1]]
        x0, x1, y0, y1 = [tf.cast(i, tf.float32) for i in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        # reshape the "big" feature map
        pixels = tf.reshape(pixels,
                            [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels,
                            [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group, channel_in])

        # copy channels to same group
        feat_in_group = self.filters // self.num_deformable_group
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, -1])

        # depth-wise conv
        out = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, filter_h, filter_w, 1], 'VALID')
        # add the output feature maps in the same group
        out = tf.reshape(out, [batch_size, out_h, out_w, self.filters, channel_in])
        out = tf.reduce_sum(out, axis=-1)
        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = inputs.get_shape().as_list()[1: 3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
                same_output = (in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           [padding_list[0], padding_list[1]],  # top, bottom padding
                           [padding_list[2], padding_list[3]],  # left, right padding
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map
        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [int(i) for i in feature_map_size[0: 2]]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_patches(images=i,
                                         sizes=[1, *self.kernel_size, 1],
                                         strides=[1, *self.strides, 1],
                                         rates=[1, *self.dilation_rate, 1],
                                         padding='VALID')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values
        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        # batch, h, w, n = y.get_shape().as_list()[0: 4]
        inShape = tf.shape(y)[0:4]
        batch = inShape[0]
        h = inShape[1]
        w = inShape[2]
        n = inShape[3]

        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)


# Capas donde se calcula el valor escalar de ICL y DCL.
# Internamente llama a funciones para su calculo, definidas mas abajo.
class ICL_fn(tf.keras.layers.Layer):
    def __init__(self, loss_weight=0.01, name='ICL_layer', **kwargs):
        super(ICL_fn, self).__init__(**kwargs)
        self.loss_weight = loss_weight

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss_weight': self.loss_weight
        })
        return config

    def call(self, inputs):
        icl_loss = get_icl_loss(inputs)
        self.add_loss(icl_loss * self.loss_weight)
        self.add_metric(icl_loss, name='ICL')
        return inputs


class DCL_fn(tf.keras.layers.Layer):
    def __init__(self, loss_weight=0.001, name='DCL_layer', **kwargs):
        super(DCL_fn, self).__init__(**kwargs)
        self.loss_weight = loss_weight

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss_weight': self.loss_weight
        })
        return config

    def call(self, inputs):
        dcl_loss = get_dcl_loss(inputs)
        self.add_loss(dcl_loss * self.loss_weight)
        self.add_metric(dcl_loss, name='DCL')
        return inputs


########################  FUNCIONES DE PERDIDA ########################################################


# Codigo adaptado a TensorFlow 2 a partir de fuente en MXNET:
# https://github.com/EigenMing/Deformable-Face-Net/blob/master/DFN_L_DCL_ICL/sym_DFN_L_DCL_ICL.py
def get_dcl_loss(inputs):
    # Computa un valor escalar para la perdida DCL.
    if inputs.shape[0] != 1:
        # Obtencion de offsets en x and y, obtenidos de la dimension de filtros
        transpose_inputs = tf.transpose(inputs, perm=[3, 0, 1, 2])  # put filters first
        partitions = tf.tile([0, 1], [9])
        offsets = tf.dynamic_partition(transpose_inputs, partitions, num_partitions=2)
        x_offset = offsets[0]
        y_offset = offsets[1]

        # Resta de offset promedio a los offsets
        x_offset_mean = tf.reduce_mean(x_offset, axis=0)  # Calculo de offset promedio de x y y
        y_offset_mean = tf.reduce_mean(y_offset, axis=0)

        x_residual = x_offset - x_offset_mean
        y_residual = y_offset - y_offset_mean

        x_residual = tf.reshape(x_residual, [-1])  # Conversion a tensor 1D
        y_residual = tf.reshape(y_residual, [-1])

        x_residual = x_residual * x_residual
        y_residual = y_residual * y_residual

        x_residual = tf.math.reduce_sum(x_residual, axis=0, keepdims=1)
        y_residual = tf.math.reduce_sum(y_residual, axis=0, keepdims=1)

        # Calculo de divisor para residuales de x y y
        dims = inputs.shape[1:-1]
        filters = inputs.shape[-1]
        divider = 1
        for dim in dims:
            divider *= dim
        divider *= filters / 2
        # En el codigo original de DFN, el divisor era was 31*31*9 = 8649
        # ( debido a que DefConv tiene salida de tamaño [31,31,18])

        # Division de residuales
        x_residual = x_residual / divider
        y_residual = y_residual / divider

        return (x_residual + y_residual)[0]
    else:
        # Si la entrada tiene tamaño de lote = 1, se evita el calculo de DCL.
        # Este es el caso para el modo de inferencia, donde las perdidas son irrelevantes.
        return tf.constant(0.0)


def get_icl_loss(inputs):
    # Computa un valor escalar para la perdida DCL.
    if inputs.shape[0] != 1:
        icl = tf.math.l2_normalize(inputs)
        # Obtencion de slices intercalados de la entrada.
        anchor = icl[0::2]
        positive = icl[1::2]

        anchor = tf.reshape(anchor, [-1])
        positive = tf.reshape(positive, [-1])

        ap = anchor - positive
        ap = ap * ap
        return tf.math.reduce_sum(ap, axis=0)
    else:
        # Si la entrada tiene tamaño de lote = 1, se evita el calculo de DCL.
        # Este es el caso para el modo de inferencia, donde las perdidas son irrelevantes.
        return tf.constant(0.0)