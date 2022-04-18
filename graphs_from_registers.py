import matplotlib.pyplot as plt
import pickle
import re
import os
import numpy as np

def load_pickle(file_dir):
    file = open(file_dir, 'rb')
    return pickle.load(file)

# Obtencion de graficas a partir de archivos pickle generados durante
# optimizacion de parametros con LRFinder
# https://github.com/beringresearch/lrfinder/blob/master/lrfinder/lrfinder.py
def plot_loss(learning_rates, losses, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
    """
    Plots the loss.
    Parameters:
        n_skip_beginning - number of batches to skip on the left.
        n_skip_end - number of batches to skip on the right.
        """
    # learning_rates = lr_finder.get_learning_rates()
    # losses = lr_finder.get_losses()

    f, ax = plt.subplots()
    ax.set_ylabel("loss")
    ax.set_xlabel("learning rate (log scale)")
    ax.plot(learning_rates[n_skip_beginning:-n_skip_end],
            losses[n_skip_beginning:-n_skip_end])
    ax.set_xscale(x_scale)

    ax.axvline(x=get_best_lr(learning_rates=learning_rates, losses=losses, sma=20,
                             n_skip_beginning=n_skip_beginning, n_skip_end=n_skip_end), c='r', linestyle='-.')

    plt.show()


def get_derivatives(learning_rates, losses, sma):
    assert sma >= 1
    derivatives = [0] * sma
    for i in range(sma, len(learning_rates)):
        derivatives.append((losses[i] - losses[i - sma]) / sma)
    return derivatives


def get_best_lr(learning_rates, losses, sma, n_skip_beginning=10, n_skip_end=5):
    derivatives = get_derivatives(learning_rates, losses, sma)
    best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
    return learning_rates[n_skip_beginning:-n_skip_end][best_der_idx]

# Grafica de perdidas a partir de exploraciones del mismo hiper parametro con diversos valores

def losses_from_subfolders(path, subfolder_name, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
    reg_compile = re.compile(subfolder_name + '-')
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if reg_compile.match(dir):
                subfolder_path = os.path.join(path, dir)
                lr_path = os.path.join(subfolder_path, 'lrs.pickle')
                loss_path = os.path.join(subfolder_path, 'losses.pickle')
                learning_rates = load_pickle(lr_path)
                losses = load_pickle(loss_path)

                plt.plot(learning_rates[n_skip_beginning:-n_skip_end],
                        losses[n_skip_beginning:-n_skip_end], label='%s' % dir)

    plt.xscale(x_scale)
    plt.title(subfolder_name)
    plt.xlabel("Learning rate")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()

# Se muestran algunos ejemplos de como se usaron tales funciones.
# A lo largo de la optimizacion de parametros se usaron muchos modelos distintos
# y se probaron valores distintos, por lo que es mas simple mostrar solo estos
# ejemplos de codigo.

'''
# Grafica de CLR
clr_loaded = load_pickle('C:\\Users\\Andrés\\Documents\\New_shard_training\\DFN_L_20_EPS_5_PATIENCE\\clr_history\\clr_history-epochs-20.pickle')
# To display entire graph
plt.plot(clr_loaded['iterations'], clr_loaded['lr'])
# To display only a portion of graph
#num_of_points = len(clr_loaded['iterations'])
#graph_start = int(num_of_points * 0.4)
#plt.plot(clr_loaded['iterations'][graph_start:], clr_loaded['lr'][graph_start:])

plt.xlabel('Training Iterations')
plt.ylabel('Learning Rate')
plt.title("One Cycle Policy")
plt.show()
'''

'''
lrs_dir = 'C:\\Users\\Andrés\\Documents\\14-12_setup\\lrs\\lrs.pickle'
loss_dir = 'C:\\Users\\Andrés\\Documents\\14-12_setup\\lrs\\losses.pickle'
#lrs_dir = 'C:\\Users\\Andrés\\Documents\\7-12_setup\\moms_15k_its\\momentum-0.95\\lrs.pickle'
#loss_dir = 'C:\\Users\\Andrés\\Documents\\7-12_setup\\moms_15k_its\\momentum-0.95\\losses.pickle'

lrs_to_plot = load_pickle(lrs_dir)
loss_to_plot = load_pickle(loss_dir)

print(lrs_to_plot[len(lrs_to_plot)-5])

plot_loss(learning_rates= lrs_to_plot, losses=loss_to_plot,
               n_skip_beginning=1000, n_skip_end=1, x_scale='log')
'''