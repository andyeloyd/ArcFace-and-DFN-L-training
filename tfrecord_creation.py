import tensorflow as tf
import glob
import os
import pickle
import time
import cv2 as cv
import pandas as pd
import tqdm


# ###### Definiciones para escritura de TFRecords ##########
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label):
    feature = {
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def paired_images_example(img1_string, img2_string, label):
    feature = {
      'label': _int64_feature(label),
      'image_1': _bytes_feature(img1_string),
      'image_2': _bytes_feature(img2_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def get_tfrec_example(img, label):
    _, img_encoding = cv.imencode('.jpg', img)
    img_encoding = img_encoding.tobytes()
    tf_example = image_example(img_encoding, label)
    return tf_example

################## Funciones de conversion a TFRecord ###############################

def convert_to_paired_tfrec(paired_list, out_path, out_file, crop_and_resize=0, image_size=112, shard_size=264800):
    #print('Starting conversion of dataset to .tfrecord format.')
    print('\t%d image pairs will be converted (%d images in total)' % (len(paired_list), len(paired_list)*2))
    print('\tEach image will be resized to a resolution of %dx%d' % (image_size, image_size))
    #print('Each shard will be comprised of %d pairs of files (%d files).' % (shard_size, shard_size * 2))
    start_time = time.time()

    write2tf_paired(paired_list, out_path, out_file, crop_and_resize=crop_and_resize,
                    size=image_size, shard_size=shard_size)
    end_time = time.time()
    print('\n\tElapsed time: %fs    Time per pair: %fs' % (end_time - start_time, (end_time - start_time) / len(paired_list) ))

def write2tf_paired(paired_list, output_path, output_filename, crop_and_resize=0, size=112, shard_size=1000):
    # crop_and_resize puede ser de 3 tipos: 0,1, or 2.
    # 0) No hace nada
    # 1) Deteccion de rostros en la imagen por MTCNN, luego recorta alrededor de la caja delimitadora y finalmente
    #    cambia el tamaño al tamaño elegido.
    # 2) Recorta levemente la imagen a lo largo del eje y, luego ajusta tamaño.
    #    Usado para cambiar imagenes de 144x122 a un tamaño cuadrado.
    print('Starting to write .tfrecord shards...')

    list_index = 0
    list_end = len(paired_list)
    shard = 0
    init_time = time.time()
    total_pairs_converted = 0

    while list_index < list_end:
        # NOTE: num_files refers to pairs of files
        num_files = 0
        record_file = os.path.join(output_path, output_filename + '_'+str(shard) + '.tfrecord')
        with tf.io.TFRecordWriter(record_file) as writer:
            for img_paths, id_name in tqdm.tqdm(paired_list[list_index:]):
                id_index = get_label_index(id_name)
                # An example in (image,label) format is obtained for each image, then each is
                # written to tfrecord sequentially
                img_1 = open_image(img_paths[0])
                img_2 = open_image(img_paths[1])
                if crop_and_resize == 1:
                    img_1 = crop_and_resize_image(img_1, size)
                    img_2 = crop_and_resize_image(img_2, size)
                elif crop_and_resize == 2:
                    trim = 22
                    img_1 = trim_y(img_1, trim)
                    img_2 = trim_y(img_2, trim)
                    img_1 = cv.resize(img_1, (size, size))
                    img_2 = cv.resize(img_2, (size, size))

                img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
                img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
                tf_example_1 = get_tfrec_example(img_1, id_index)
                tf_example_2 = get_tfrec_example(img_2, id_index)

                writer.write(tf_example_1.SerializeToString())
                writer.write(tf_example_2.SerializeToString())
                num_files = num_files + 1

                total_pairs_converted += 1

                if num_files >= shard_size or num_files + list_index == list_end:
                    list_index = list_index + num_files
                    print('\nList_index: %d' % list_index)
                    break

        shard = shard + 1
        end_of_shard_time = time.time()
        elapsed_time = (end_of_shard_time - init_time)
        print('\nShard %d finished after %fs / %f min / %f hours.' % (shard, elapsed_time, elapsed_time/60, elapsed_time/3600))
        print('\nTotal pairs converted: %d - (%d files' % (total_pairs_converted, total_pairs_converted*2))

# ############# Image handling functions ########################


def open_image(img_path):
    #print(img_path)
    return cv.imread(img_path)


def crop_square_face(img, bbox_coordinates):
    # Convert to int
    bbox_coordinates = [abs(int(i)) for i in bbox_coordinates]
    # If any value is negative, it becomes zero instead
    bbox_coordinates = remove_negatives(bbox_coordinates)
    # Make bbox square
    bbox_coordinates = rectify_bbox(bbox_coordinates)
    # Once again, eliminate negative coordinates
    bbox_coordinates = remove_negatives(bbox_coordinates)
    y1, x1, y2, x2 = bbox_coordinates
    # Perform the cropping on the input image
    roi = img[y1:y2, x1:x2]
    return roi


def remove_negatives(bbox_coordinates):
    for i in range(len(bbox_coordinates)):
        if bbox_coordinates[i] < 0:
            bbox_coordinates[i] = bbox_coordinates[i]*(-1)
    return bbox_coordinates


def rectify_bbox(bbox):
    y1, x1, y2, x2 = bbox
    y = y2-y1
    x = x2-x1
    if x > y:
        delta = (x-y) / 2
        y1 = int(y1 - delta)
        y2 = int(y2 + delta)
    elif y > x:
        delta = (y-x) / 2
        x1 = int(x1 - delta)
        x2 = int(x2 + delta)
    return [y1, x1, y2, x2]


def crop_and_resize_image(img, size=112, min_size=30, factor=0.7, thresholds=[0.6, 0.7, 0.8]):
    # Detect faces on the input image
    detections = mtcnn_fun(img, min_size, factor, thresholds)
    # If any face is found, the image is cropped around the face with highest confidence.
    if len(detections[0]) > 0:
        img = crop_square_face(img, detections[0][0])
    # Perform the resizing of the image
    img = cv.resize(img, (size, size))
    return img


def trim_y(img, total_trim):
    h, w, _ = img.shape
    trim = int(total_trim/2)
    img = img[trim:(h-trim), 0:w]
    return img

##################### Detector de rostros ############################################
def mtcnn_fun(img, min_size, factor, thresholds):
    with open('./mtcnn.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef.FromString(f.read())

    with tf.device('/cpu:0'):
        prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
            input_map={
                'input:0': img,
                'min_size:0': min_size,
                'thresholds:0': thresholds,
                'factor:0': factor
            },
            return_elements=[
                'prob:0',
                'landmarks:0',
                'box:0']
            , name='')
    #print(box, prob, landmarks)
    return box, prob, landmarks

# ######## Funciones de metadatos de CSV ################


def get_namelist_and_classdict(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    name_array = df['Name'].to_numpy()
    class_dict = dict(zip(df['Class_ID'], range(0, 9131)))
    return name_array, class_dict

def get_label_index(label_string):
    return classDict[label_string]


####################### Funciones de carga de lista ###########################
def get_pickle_list(path, sorted=True):
    pickle_list = glob.glob(os.path.join(path, '*.pickle'))
    print('Found %d matching files.' % len(pickle_list))
    pickle_list.sort()
    return pickle_list

def load_list_from_pickle(file_path):
    """'
    file = open(file_path, 'r')
    data = file.read().split('\n')
    lines = [line for line in data]
    file.close()
    lines = lines[:(len(lines) - 1)]
    print('Loaded a file list containing %d elements.' % len(lines))
    """
    print('Loading a file list from %s...' % file_path)
    with open(file_path, "rb") as f:
        list = pickle.load(f)
    print('Loaded a file list containing %d elements.' % len(list))
    return list

# Metadatos de VGGFace2
# 8631 training ids
#dataset_csv = os.path.abspath('C:\\Users\\Andrés\\Downloads\\identity_meta_train.csv')
# 500 test ids
dataset_csv = os.path.abspath('C:\\Users\\Andrés\\Downloads\\identity_meta_test.csv')
nameArray, classDict = get_namelist_and_classdict(dataset_csv)


# Inicializacion de detector de rostros para cortado de ellos
mtcnn_fun = tf.compat.v1.wrap_function(mtcnn_fun, [
    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[3], dtype=tf.float32)
])


# Fragmentos de conjunto de entrenamiento de VggFace2
#pickle_path = 'C:\\Users\\Andrés\\Documents\\vggface2\\split_file_lists'
# Fragmentos de conjunto de prueba de VggFace2
split_lists = ['C:\\Users\\Andrés\\Documents\\vggface2_test_shards\\file_list\\train.pickle']


#split_lists = get_pickle_list(path=pickle_path)
print(split_lists)


#train_path_list = load_list_from_pickle('D:\\Datasets\\tfrec_vgg2_clean_122x122\\new_splits\\file_lists\\vggface2_test.pickle')
#val_path_list = load_list_from_pickle("D:\\Datasets\\tfrec_vgg2_clean_122x122\\val.pickle")

train_out_file = 'vggface2_test_trainDS_224x224'
#train_out_path = 'C:\\Users\\Andrés\\Documents\\vggface2_test_shards\\shards'
train_out_path = "C:\\Users\\Andrés\\Documents\\throwaway\\fake_shards"
if not os.path.isdir(train_out_path):
    os.makedirs(train_out_path)

#crop_and_resize_method = 0  # No hace nada
crop_and_resize_method = 1  # Detectar y recortar rostros, luego redimensionar.
#crop_and_resize_method = 2  # Cambia imagenes de 144x122 a tamaño cuadrado, luego redimensiona.
image_size = 224

split_count = 0
for split_file_list in split_lists:
    path_list = load_list_from_pickle(split_file_list)
    shard_train_out_file = train_out_file + str(split_count)
    recs_by_shard = len(path_list)
    print('Converting to TFRecord: %s' % split_file_list)
    print('List contains %d pairs.' % len(path_list))
    convert_to_paired_tfrec(path_list, train_out_path, shard_train_out_file,
                            crop_and_resize=crop_and_resize_method, image_size=image_size, shard_size=recs_by_shard)
    split_count += 1
    print('Converted %d shards so far.' % split_count)
