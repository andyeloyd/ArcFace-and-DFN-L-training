import tensorflow as tf
import pandas as pd
import os
import pickle
import math

from model_definition import get_dfn_l_model
from dataset_loading import dataset_from_shards
from new_callbacks import CyclicLR

def save_to_pickle(data, save_path, file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_dir = os.path.join(save_path, file_name)
    with open(save_dir, 'wb') as file:
        pickle.dump(data, file)
    print('Saved losses to : %s' % save_dir)

def training_info(epochs, batch_size, num_images_dataset, iterations_per_epoch, iterations,
                  num_classes, m, s, maximum_lr, minimum_lr, max_m, base_m, wd):
    print('Starting training process...')
    print('The configuration is the following:')
    print('EPOCHS: %d' % epochs)
    print('Batch size: %d' % batch_size)
    print('Dataset images: %d' % num_images_dataset, end=',')
    print(' resulting in %d iterations per epoch and %d iterations overall' % (iterations_per_epoch, iterations))
    print('Hyperparameters:')
    print('NUM_CLASSES= %d, M=%0.2f, S=%0.1f' % (num_classes, m, s))
    print('Max_lr: %0.4f, Min_lr: %0.4f' % (maximum_lr, minimum_lr))
    print('Max_momentum: %0.2f, Min_momentum: %0.2f' % (max_m, base_m))
    print('Weight decay: %0.5f' % wd)

#####################----CARGA DE METADATOS DE CONJUNTOS DE DATOS----######################################################

vgg2Meta_csv = os.path.abspath("C:\\Users\\Andrés\\Downloads\\identity_meta_train.csv")
vgg2MetaDF = pd.read_csv(vgg2Meta_csv, skipinitialspace=True)
# For speeding up look-up times (faster than in a csv)
nameArray = vgg2MetaDF['Name'].to_numpy()
classDict = dict(zip(vgg2MetaDF['Class_ID'], range(0, 8631)))


######################----DEFINICION DE PARAMETROS----#########################################################
epochs = 30
batch_size = 64
num_class = 8631

# Parametro m de ArcFace
m = 0.3
# Parametro s de ArcFace, calculado segun formula de AdaCos
s = math.sqrt(2)*math.log(num_class - 1)

# Parametros para OCP
wd = 1e-4
maximum_lr = 0.01
minimum_lr = maximum_lr / 10
max_m = 0.95
base_m = 0.8
cyclical_momentum = True
cycles = 2.35

# CONJUNTO DE ENTRENAMIENTO: 29x 50718 pares + 1x 50712 pares a lo largo de los fragmentos
# CONJUNTO DE VALIDACION: 1x 24105 pares a lo largo de los fragmentos
# EN TOTAL: 3,091,278 IMAGENES
num_images_dataset = 2 * (50718 * 29 + 50712)
iterations_per_epoch = num_images_dataset / batch_size
iterations = iterations_per_epoch * epochs
step_size = iterations / cycles

##########----CARGA DE CONJUNTO DE DATOS----################################################
tfrecords_dir = os.path.abspath('C:\\Users\Andrés\\Documents\\vggface2\\new_splits\\train_split_shards')
train_tfrec_name = 'new_trainDS_pairs_122x122'
val_tfrec_name = 'new_valDS_pairs_122x122'

train_dataset = dataset_from_shards(tfrecords_dir, train_tfrec_name, batch_size=batch_size)
val_dataset = dataset_from_shards(tfrecords_dir, val_tfrec_name, batch_size=batch_size)


#####################----CALLBACKS----###################################################################

# Definicion de directorios donde se alojaran los registros

# Directorio de etapa con OCP
# master_save_path = 'C:\\Users\\Andrés\\Documents\\New_shard_training\\DFN_L_20_EPS_5_PATIENCE'
# Directorio de etapa post-OCP
master_save_path = 'C:\\Users\\Andrés\\Documents\\New_shard_training\\DFN_L_20_EPS(POST_OCP)'


chkpts_path = os.path.join(master_save_path, 'checkpoints')
backup_filepath = os.path.join(master_save_path, 'training_backup')
if not os.path.isdir(backup_filepath):
    os.makedirs(backup_filepath)
backup_filepath = os.path.join(backup_filepath, 'backup_file')
chkpts_filename = os.path.join(chkpts_path, 'weights-{epoch:03d}-{val_loss:.4f}.hdf5')
tensorboard_logdir = os.path.join(master_save_path, 'tensorboard')
clr_history_filepath = os.path.join(master_save_path, 'clr_history\\clr_history_epoch-%s.pickle' % epochs)

# Inicializacion de callbacks
callback_tensorboard = tf.keras.callbacks.TensorBoard(tensorboard_logdir, histogram_freq=1)
callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01,
                                                       patience=5, mode='min')
clr = CyclicLR(base_lr=minimum_lr, max_lr=maximum_lr, step_size=step_size,
               max_m=max_m, base_m=base_m, cyclical_momentum=cyclical_momentum)
callback_checkpoints = tf.keras.callbacks.ModelCheckpoint(chkpts_filename, monitor='val_loss',
                                                          verbose=1, save_best_only=True)
callback_backup = tf.keras.callbacks.experimental.BackupAndRestore(backup_filepath)

# Callbacks durante OCP
#callback_list = [callback_tensorboard, callback_early_stop, clr, callback_checkpoints, callback_backup]
# Callbacks durante post-OCP (sin CLR)
callback_list = [callback_tensorboard, callback_early_stop, callback_checkpoints, callback_backup]


####################### INICIALIZACION Y COMPILADO DEL MODELO ################################################

model = get_dfn_l_model(size=122, channels=3, embd_shape=512,
                        dcl_weight=0.001, icl_weight=0.01, wd=wd,
                        num_classes=num_class, m=m, s=s, head='dense', name='dfn_l')
#model.summary()

# Cargar pesos pre-entrenados por OCP
pretrained_weights_path = 'C:\\Users\\Andrés\\Documents\\New_shard_training\\DFN_L_20_EPS_5_PATIENCE\\saved_weights\\saved_weights.h5'
model.load_weights(pretrained_weights_path)


# Durante etapa con OCP
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=max_m, nesterov=True)
# Post-OCP
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics='accuracy')

# Despliegue de valores de hiper parametros
training_info(epochs=epochs, batch_size=batch_size, num_images_dataset=num_images_dataset,
              iterations_per_epoch=iterations_per_epoch, iterations=iterations, num_classes=num_class,
              m=m, s=s, maximum_lr= maximum_lr, minimum_lr=minimum_lr, max_m=max_m, base_m=base_m, wd=wd)
print('Relevant files and logs will be saved in: %s' % (master_save_path))

# Entrenamiento de red
model.fit(x=train_dataset, epochs=epochs, verbose=1, callbacks=callback_list, validation_data=val_dataset)

# Guardado de historial de CLR a disco
pickle_path = os.path.join(master_save_path, 'clr_history')
if not os.path.isdir(pickle_path):
    os.makedirs(pickle_path)
save_to_pickle(clr.history, pickle_path, 'clr_history-epochs-%s.pickle' % epochs)

# Guardado de los pesos entrenados a disco

weights_save_dir = os.path.join(master_save_path, 'saved_weights')
if not os.path.isdir(weights_save_dir):
    os.makedirs(weights_save_dir)
weights_save_dir = os.path.join(weights_save_dir, 'saved_weights.h5')
print('Saved model weights to %s' %weights_save_dir)

model.save(weights_save_dir)