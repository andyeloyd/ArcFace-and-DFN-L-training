import glob
import tqdm
import math
import random
import numpy as np
import pandas as pd
import pickle
import os


######################### Funciones de creacion de lista ############################

def make_pairs(img_number):
    # Split indexes in two lists
    array_limit = int(math.ceil(img_number / 2))
    first_list = [np.arange(array_limit)]
    second_list = [np.arange(array_limit, img_number)]
    # If number of images is odd, an extra image is copied to complete pairs
    if (img_number % 2) != 0:
        second_list = np.hstack((second_list, [[random.choice(second_list[0])]]))
    # Make paired list
    first_list = np.swapaxes(first_list, 1, 0)
    second_list = np.swapaxes(second_list, 1, 0)
    third_list = np.append(first_list, second_list, axis=1)

    return third_list


def substitute_in_list(indexes_list, source_list):
    # replaces a int elements of a 2D list with the corresponding index of a 1D list
    rows, cols = np.shape(indexes_list)
    indexes_list = indexes_list.tolist()
    for row in range(rows):
        for col in range(cols):
            indexes_list[row][col] = source_list[indexes_list[row][col]]
    return indexes_list


def pairwise_img_list_from_dataset(dataset_path, id_limit=None, shuffle_id_order=False,
                                   shuffle_before_pairing=False, shuffle_pairs=False, val_split=None):
    print('\nA list of paired images will be made from the following path:', end=' ')
    print("'" + dataset_path + "'")
    if val_split:
        print('A validation split of %.2f percent will be made from the given dataset' % (val_split*100))
    if shuffle_id_order:
        print('Order of the subfolders will be shuffled')
    if shuffle_before_pairing:
        print('Order of images inside each subfolder will be shuffled')
    print('Making paired image lists...')
    subfolders = os.listdir(dataset_path)
    if shuffle_id_order:
        random.shuffle(subfolders)
    else:
        subfolders.sort()
    if id_limit:
        subfolders = subfolders[:id_limit]

    train_samples = []
    val_samples = []
    for id_name in tqdm.tqdm(subfolders):
        img_paths = glob.glob(os.path.join(dataset_path, id_name, '*.jpg'))
        if shuffle_before_pairing:
            # shuffles order of images in subfolder
            random.shuffle(img_paths)
        else:
            img_paths.sort()
        id_imgs = len(img_paths)
        img_list = make_pairs(id_imgs)
        img_list = substitute_in_list(img_list, img_paths)
        if val_split:
            split_point = (1-val_split)*len(img_list)
            # Round down to nearest even number
            split_point = int(split_point/2)*2
            train_list = img_list[:split_point]
            val_list = img_list[split_point:]

            for img_pair in train_list:
                train_samples.append((img_pair, id_name))
            for img_pair in val_list:
                val_samples.append((img_pair, id_name))
        else:
            for img_pair in img_list:
                train_samples.append((img_pair, id_name))

    print('\n\nLength of file lists:')
    print('\tTraining: %d pairs (%d total images)' % (len(train_samples), len(train_samples)*2))
    if val_split:
        print('\tValidation: %d pairs (%d total images)' % (len(val_samples), len(val_samples)*2))

    if shuffle_pairs:
        random.shuffle(train_samples)
        random.shuffle(val_samples)
    return train_samples, val_samples

################ Funciones de guardado y fragmentado de lista #############################

def save_img_list(list, filename, path):
    '''
    out_path = os.path.join(path, filename)
    with open(out_path, 'w') as output:
        for img in list:
            output.write(str(img)+'\n')
    '''
    out_path = os.path.join(path, filename)
    with open(out_path, 'wb') as f:
        pickle.dump(list, f, pickle.HIGHEST_PROTOCOL)
    print('Saved a file to the following path: %s' % out_path)

def load_list(file_path):
    with open(file_path, "rb") as f:
        dump = pickle.load(f)
    return dump


def write_to_pickle(list,file_path):
    with open(file_path, "wb") as f:
        pickle.dump(list, f, pickle.HIGHEST_PROTOCOL)


def split_list(original_list, split_number):
    new_list = []
    split_size = math.ceil(len(original_list) / split_number)
    list_start = 0

    for i in range(split_number):
        #list_start = split_size*i
        list_end = list_start + split_size
        if list_end > len(original_list):
            list_end = len(original_list)
        new_list.append(original_list[list_start:list_end])
        list_start += split_size
    return new_list


def split_list_to_pickle(new_list, file_name, save_path):
    i = 0
    for line in new_list:
        new_file_name = file_name + str(i) + '.pickle'
        shard_name = os.path.join(save_path, new_file_name)
        print('Saving pickle file to: %s' % shard_name)
        print('(Contains %d image pairs (%d images))' % (len(line), 2*len(line)))
        write_to_pickle(line, shard_name)
        i += 1
    print('Wrote %d pickle files in total.' % i)





# Directorio de conjunto de datos
#ds_path = os.path.abspath('C:\\vggface2\\VGGFACE2_Cleandata\\train')
ds_path = os.path.abspath('C:\\vggface2_test')

id_limit = None
val_split = 0.2
save_path = 'C:\\Users\\Andrés\\Documents\\vggface2_test_shards\\file_list'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
train_path_list, val_path_list = pairwise_img_list_from_dataset(ds_path, id_limit=id_limit, shuffle_id_order=True,
                                                                shuffle_before_pairing=True, shuffle_pairs=True, val_split=val_split)


#Divison de lista de entrenamiento en lista para cada fragmento
split_number = 30
new_list = split_list(train_path_list, split_number)

print('Fragment number: %d' % len(new_list))
print('Number of pairs within each fragment:')
summed_length = 0
for i in range(len(new_list)):
    print('Fragment %d pairs: %d' % (i,len(new_list[i])))
    summed_length += len(new_list[i])
print('Total pairs: %d' %summed_length)


save_path = "C:\\Users\\Andrés\\Documents\\vggface2\\split_file_lists"
if not os.path.isdir(save_path):
    os.makedirs(save_path)
file_name = "new_train_split_"

split_list_to_pickle(new_list, file_name, save_path)

save_img_list(val_path_list, 'val.pickle', save_path)
