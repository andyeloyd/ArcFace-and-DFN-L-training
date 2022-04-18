import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os


image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def parse_img(feature):
    # img = tf.io.read_file(feature['image_raw'])
    img = feature['image_raw']
    return tf.cast(tf.image.decode_jpeg(img), tf.float32)


def map_img_color(img):
    img_unstacked = tf.unstack(img, axis=-1)
    img = tf.stack([img_unstacked[2], img_unstacked[1], img_unstacked[0]], axis=-1)
    img = img / 255.0
    return img

def parse_img_label(example):
    parsed = _parse_image_function(example)
    img = parse_img(parsed)
    img = tf.ensure_shape(img, (122,122,3))
    img = map_img_color(img)
    label = parsed['label']

    return img, label

def data_aug(x, y):
  img = x
  img = tf.image.random_flip_left_right(img)
  img = tf.image.per_image_standardization(img)
  return img, y

def img_rotation(x,y):
  angle = np.random.uniform(low=-np.pi/18, high=np.pi/18)
  x = tfa.image.rotate(x, angle, fill_mode='nearest')
  return x, y

def data_formatting(x,y):
  return (x,y),y


def dataset_from_shards(tfrecords_dir, tfrec_name, batch_size=64, rotation=False):
    """

    :param tfrecords_dir: Directorio donde se alojan los fragmentos de TFRecords a usar.
    :param tfrec_name:    Nombre en comun por el cual identificar los fragmentos a usar.
    :param batch_size:    Tamaño de lote que debe tener el conjunto de datos.
    :return:              Objeto tf.data.dataset, barajado por pares (Cada 2 imagenes pertenecen
                          a la misma identidad. Esto es requerido para entrenar DFN)
    """

    file_dir = os.path.join(tfrecords_dir, '%s_*' % tfrec_name)
    files = tf.io.matching_files(file_dir)
    shard_list = tf.data.Dataset.from_tensor_slices(files)

    # Carga de TFRecords
    dataset = shard_list.interleave(lambda x: tf.data.TFRecordDataset(x),
                                    cycle_length=4,
                                    block_length=2,
                                    num_parallel_calls=tf.data.AUTOTUNE,
                                    deterministic=False)
    # NOTA: cycle length es el numero de elementos de entrada a procesar simultaneamente,
    #       block_length es el numero de elementos consecutivos a cargar por cada fuente. (2)

    # Barajar datos por pares
    dataset = dataset.batch(2)
    dataset = dataset.shuffle(buffer_size=16384)  # 2^14
    dataset = dataset.unbatch()

    # Parsing de datos (conversiona formato imagen)
    dataset = dataset.map(parse_img_label)
    # Aplicacion de aumento de datos
    if rotation:
        dataset = dataset.map(img_rotation)
    dataset = dataset.map(data_aug)
    # Acomodo de datos a formato requerido para ArcFace
    dataset = dataset.map(data_formatting)

    # Formacion de lotes
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset