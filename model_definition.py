import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation

from tensorflow.keras import regularizers

from custom_layers_and_losses import *

# Functiones para agregar etapas al modelo

# Etapa de entrada y primer convolucion
def stage_0(x, conv_filters=64, kernel=(7, 7), strides=4, wd=None):
    if wd:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(wd)

    x = BatchNormalization(name='BN_00')(x)
    x = Conv2D(filters=conv_filters, kernel_size=kernel, strides=strides, padding='same',
               kernel_regularizer=kernel_regularizer, name='Conv_0')(x)
    x = Activation("relu", name='ReLu_0')(x)
    x = BatchNormalization(name='BN_01')(x)
    return x


# Etapa 1
def stage_1(x, pool_size=(3, 3), strides=2):
    x = MaxPooling2D(pool_size, strides, name='MaxPool', padding='same')(x)
    x = Activation("relu", name='ReLu_1')(x)
    x = BatchNormalization(name='BN_1')(x)
    return x


# Modulo deformable
def deformable_module_fn(x, conv_filters=64, kernel=(3, 3), strides=1, dcl_weight=0.001, icl_weight=0.01, wd=None):
    if wd:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(wd)

    disp_field = Conv2D(18, (3, 3), padding='same', name='Displacement_field')(x)
    x = DCL_fn(loss_weight=dcl_weight, name='DCL')(disp_field)
    x = DeformableConvLayer(conv_filters, kernel, strides, padding='same', num_deformable_group=1,
                            kernel_regularizer=kernel_regularizer, name='Def_Conv')(x)
    x = ICL_fn(loss_weight=icl_weight, name='ICL')(x)
    return x


# Etapa(s) intermedias
def middle_stage(x, conv_filters, kernel, strides, stage_num, wd=None, padding='same'):
    if wd:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(wd)

    x = Activation("relu", name=('ReLu_' + str(stage_num)))(x)
    x = BatchNormalization(name=('BN_' + str(stage_num)))(x)
    x = Conv2D(filters=conv_filters, kernel_size=kernel, strides=strides, padding=padding,
               kernel_regularizer=kernel_regularizer, name=('Conv_' + str(stage_num)))(x)
    return x


# Etapa de capa embebida
def embed_stage(x, embd_shape, pool_size=(4, 4), wd=None):
    if wd:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(wd)

    x = Activation("relu", name='ReLu_FC')(x)
    x = BatchNormalization(name='BN_FC')(x)
    x = AveragePooling2D(pool_size, padding='valid', name='Avg_Pool')(x)
    x = Flatten(name='Flatten')(x)
    x = Dense(embd_shape, kernel_regularizer=kernel_regularizer, name='FC_Embedding')(x)
    return x


# Funcion de inicializacion de modelo DFN-L
def get_dfn_l_model(size=None, channels=3, embd_shape=512, dcl_weight=0.001,
                    icl_weight=0.01, wd=None, num_classes=None, m=0.5, s=64, head=None, name='model_with_head'):
    """
    :param size:        Tamaño de entrada del modelo.
    :param channels:    Numero de canales de las imagenes de entrada.
    :param embd_shape:  Tamaño de dimensiones para capa embebida.
    :param dcl_weight:  Peso de la perdida DCL.
    :param icl_weight:  Peso de la perdida ICL.
    :param wd:          Valor de la decaida de pesos.
    :param num_classes: Numero de clases a clasificar.
    :param m:           Parametro m de ArcFace (margen). Se ignora si no se usa ArcFace.
    :param s:           Parametro s de ArcFace (escala). Se ignora si no se usa ArcFace.
    :param head:        Tipo de capa final (Dense o ArcFace)
    :param name:        Nombre a asignarle al modelo.
    :return:            Modelo DFN-L con los hiper parametros dados (sin compilar o entrenar aun).
    """
    if wd:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(wd)

    inputs = Input(shape=[size, size, channels], name='model_input')
    y = Input([], name='label')

    x = stage_0(inputs, conv_filters=64, kernel=(7, 7), strides=2, wd=wd)
    x = stage_1(x)
    x = deformable_module_fn(x, dcl_weight=dcl_weight, icl_weight=icl_weight)
    x = middle_stage(x, conv_filters=64, kernel=3, strides=2, wd=wd, stage_num=2)
    x = middle_stage(x, conv_filters=128, kernel=3, strides=2, wd=wd, stage_num=3)
    x = middle_stage(x, conv_filters=128, kernel=3, strides=1, wd=wd, stage_num=4)
    x = embed_stage(x, embd_shape, (3, 3))

    if head.lower() == 'arcface':
        x = Activation("relu", name='ReLu_Final')(x)
        x = BatchNormalization(name='BN_Final')(x)
        x = ArcFace(n_classes=num_classes, s=s, m=m, regularizer=kernel_regularizer)(x, y)
        return Model((inputs, y), x, name=name)
    elif head.lower() == 'dense':
        x = Activation("relu", name='ReLu_Final')(x)
        x = BatchNormalization(name='BN_Final')(x)
        x = Dense(num_classes, activation=tf.keras.activations.softmax, name='FC_Classes')(x)
        return Model((inputs, y), x, name=name)
    else:
        print('Need other head.')
        return
