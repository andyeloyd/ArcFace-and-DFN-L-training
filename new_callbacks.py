import tensorflow as tf
import math
import numpy as np
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

# LRFinder: Callback para optimizacion de parametros
# Implementacion tomada de:
# https://github.com/beringresearch/lrfinder/blob/master/lrfinder/lrfinder.py
class LRFinder_2:
    """
    Learning rate range test detailed in Cyclical Learning Rates for Training
    Neural Networks by Leslie N. Smith. The learning rate range test is a test
    that provides valuable information about the optimal learning rate. During
    a pre-training run, the learning rate is increased linearly or
    exponentially between two boundaries. The low initial learning rate allows
    the network to start converging and as the learning rate is increased it
    will eventually be too large and the network will diverge.
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.learning_rates = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

        loss = logs['loss']
        self.losses.append(loss)

        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, dataset, start_lr, end_lr, epochs=1,
             steps_per_epoch=None, **kw_fit):
        if steps_per_epoch is None:
            raise Exception('To correctly train on the datagenerator,'
                            '`steps_per_epoch` cannot be None.'
                            'You can calculate it as '
                            '`np.ceil(len(TRAINING_LIST) / BATCH)`')

        self.lr_mult = (float(end_lr) /
                        float(start_lr)) ** (float(1) /
                                             float(epochs * steps_per_epoch))
        initial_weights = self.model.get_weights()

        original_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch,
                                  logs: self.on_batch_end(batch, logs))
        # modified to run only for a given number of steps per epoch
        #self.model.fit(dataset,
        #               epochs=epochs, callbacks=[callback], **kw_fit)
        self.model.fit(dataset,
                       epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[callback], **kw_fit)

        self.model.set_weights(initial_weights)

        K.set_value(self.model.optimizer.lr, original_lr)

    def get_learning_rates(self):
        return(self.learning_rates)

    def get_losses(self):
        return(self.losses)

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.learning_rates)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.learning_rates[n_skip_beginning:-n_skip_end][best_der_idx]



# Callback para usar One Cycle Policy (OCP).
# Implementacion tomada de:
# https://www.kaggle.com/robotdreams/one-cycle-policy-with-keras/notebook
class CyclicLR(Callback):

    def __init__(self, base_lr, max_lr, step_size, base_m, max_m, cyclical_momentum,
                 clr_iterations=0., cm_iterations=0., trn_iterations=0., history={}):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.base_m = base_m
        self.max_m = max_m
        self.cyclical_momentum = cyclical_momentum
        self.step_size = step_size

        self.clr_iterations = clr_iterations
        self.cm_iterations = cm_iterations
        self.trn_iterations = trn_iterations
        self.history = history

    def clr(self):

        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))

        if cycle == 2:
            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
            return self.base_lr - (self.base_lr - self.base_lr / 100) * np.maximum(0, (1 - x))

        else:
            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))

    def cm(self):

        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))

        if cycle == 2:

            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
            return self.max_m

        else:
            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
            return self.max_m - (self.max_m - self.base_m) * np.maximum(0, (1 - x))

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

        if self.cyclical_momentum == True:
            if self.clr_iterations == 0:
                K.set_value(self.model.optimizer.momentum, self.cm())
            else:
                K.set_value(self.model.optimizer.momentum, self.cm())

    def on_batch_begin(self, batch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        if self.cyclical_momentum == True:
            self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

        if self.cyclical_momentum == True:
            K.set_value(self.model.optimizer.momentum, self.cm())