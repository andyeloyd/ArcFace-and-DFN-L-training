import tensorflow as tf
import os
import math
import pickle

from model_definition import get_dfn_l_model
from dataset_loading import dataset_from_shards
#from new_callbacks import LRFinder_2, CyclicLR
from custom_callbacks import  LRFinder_2

def save_to_pickle(data, save_path, file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_dir = os.path.join(save_path, file_name)
    with open(save_dir, 'wb') as file:
        pickle.dump(data, file)
    print('Saved losses to : %s' % save_dir)


num_images_dataset = 2 * (50718 * 29 + 50712)
num_class = 8631
epochs = 1
batch_size = 64

# Parametros preliminares antes de optimizacion
wd = 1e-4
maximum_lr = 1
minimum_lr = 1e-4
momentum = 0.9

# Parametros de ArcFace
m = 0.3
s = math.sqrt(2)*math.log(num_class - 1)


##########----CARGA DE CONJUNTO DE DATOS----################################################
tfrecords_dir = os.path.abspath('C:\\Users\Andrés\\Documents\\vggface2\\new_splits\\train_split_shards')
train_tfrec_name = 'new_trainDS_pairs_122x122'
train_dataset = dataset_from_shards(tfrecords_dir, train_tfrec_name, batch_size=batch_size)

# Se deben obtener los valores de tasa de aprendizaje primero, despues, momento y finalmente
# decaida de peso. Comentar secciones que no seran usadas, y ajustar valores segun lo requerido.



############################ Exploracion de rango de LR. #########################################
num_samples = num_images_dataset
maximum_lr = 1
minimum_lr = 1e-4
save_dir = 'C:\\Users\\Andrés\\Documents\\arc_dfn_s_auto_500_ids'
steps_per_epoch = num_samples / batch_size

print('Exploring LR values across ', end='')
print('%d epochs.' % epochs)
print('Steps per epoch: %d' % steps_per_epoch)
print('Max. learning rate: %0.4f \nMin. learning rate: %0.4f' % (maximum_lr, minimum_lr))
print('NUM_CLASSES= %d, M=%0.2f, S=%0.1f' % (num_class, m, s))
print('Save directory: %s' %save_dir)

model = get_dfn_l_model(size=122, channels=3, embd_shape=512,
                        dcl_weight=0.001, icl_weight=0.01, wd=wd,
                        num_classes=num_class, m=m, s=s, head='dense', name='dfn_l')
# LR no importa, pues sera modificado por LRFinder
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=momentum, nesterov=True)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics='accuracy')
lr_finder_bering= LRFinder_2(model)
lr_finder_bering.find(train_dataset, minimum_lr, maximum_lr, epochs=1, steps_per_epoch=steps_per_epoch)

lrs = lr_finder_bering.get_learning_rates()
losses = lr_finder_bering.get_losses()

save_to_pickle(lrs, os.path.join(save_dir, 'lrs'), 'lrs.pickle')
save_to_pickle(losses, os.path.join(save_dir, 'lrs'), 'losses.pickle')


'''
############################ Exploracion de valores de momento #########################################
save_dir = 'C:\\Users\\Andrés\\Documents\\arc_dfn_s_auto_500_ids\\momentums'
# Rango de tasas de aprendizaje (ya optimizadas)
maximum_lr = 0.01
minimum_lr = maximum_lr / 10

#Valores a explorar
MOMENTUMS = [0.8 ,0.85, 0.9, 0.95, 0.99]
steps_per_epoch = num_images_dataset / batch_size
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


print('Exploring MOMENTUM values across ', end='')
print('%d epochs.' % epochs)
print('Steps per epoch: %d' % steps_per_epoch)
print('Max. learning rate: %0.4f \nMin. learning rate: %0.4f' % (maximum_lr, minimum_lr))
print('Momentum factors: [', end='')


for momentum in MOMENTUMS:
    print(' ', end='')
    print(momentum, end=' ')
print(']')
print('NUM_CLASSES= %d, M=%0.2f, S=%0.1f' % (num_class, m, s))
print('Save directory: %s' %save_dir)


for momentum in MOMENTUMS:
    print('TRYING OUT MOMENTUM: %0.5f' % momentum)
    model = get_dfn_l_model(size=122, channels=3, embd_shape=512,
                            dcl_weight=0.001, icl_weight=0.01, wd=wd,
                            num_classes=num_class, m=m, s=s, head='arcface', name='dfn_arcface')
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=momentum, nesterov=True)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics='accuracy')

    lr_finder_bering = LRFinder_2(model)
    lr_finder_bering.find(train_dataset, minimum_lr, maximum_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    lrs = lr_finder_bering.get_learning_rates()
    losses = lr_finder_bering.get_losses()

    save_to_pickle(lrs, os.path.join(save_dir, 'momentum-%s' % momentum), 'lrs.pickle')
    save_to_pickle(losses, os.path.join(save_dir, 'momentum-%s' % momentum), 'losses.pickle')
'''


'''

################## Exploracionde weight decay #############################################
save_dir = 'C:\\Users\\Andrés\\Documents\\dfn_l\\weight_decay_full_epoch'
steps_per_epoch = (num_images_dataset / batch_size)
momentum = 0.9

WD_FACTORS = [0, 1e-5, 5e-4, 1e-4, 1e-3]

print('Exploring weight decay values across ', end='')
print('%d epochs.' % epochs)
print('Steps per epoch: %d' % steps_per_epoch)
print('Max. learning rate: %0.4f \nMin. learning rate: %0.4f' % (maximum_lr, minimum_lr))
print('Decay factors: [', end='')
for wd in WD_FACTORS:
    print(' ', end='')
    print(wd, end=' ')
print(']\nSave directory: %s' %save_dir)

for wd in WD_FACTORS:
    print('TRYING OUT WD: %0.5f' %wd)
    model = get_dfn_l_model(size=122, channels=3, embd_shape=512,
                            dcl_weight=0.001, icl_weight=0.01, wd=wd,
                            num_classes=num_class, m=m, s=s, head='dense', name='dfn_l')
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=momentum, nesterov=True)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics='accuracy')

    lr_finder_bering = LRFinder_2(model)
    lr_finder_bering.find(train_dataset, minimum_lr, maximum_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    lrs = lr_finder_bering.get_learning_rates()
    losses = lr_finder_bering.get_losses()

    save_to_pickle(lrs, os.path.join(save_dir, 'weight_decay-%s' % wd), 'lrs.pickle')
    save_to_pickle(losses, os.path.join(save_dir, 'weight_decay-%s' % wd), 'losses.pickle')
'''