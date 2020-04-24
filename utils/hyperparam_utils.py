import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import keras 

def get_optimizer(optimizer, learning_rate=1e-4):
    '''
    '''
    #TODO: For finetuning, more parameters (e.g momentum params) could be tuned like beta1 and beta2

    if optimizer == 'RMSProp': 
        optimizer_configured = keras.optimizers.RMSprop(learning_rate=learning_rate)

    elif optimizer == 'SGD': 
        optimizer_configured = keras.optimizers.SGD(learning_rate=learning_rate / 10, decay=3e-6)

    else: 
        optimizer_configured = keras.optimizers.Adam(learning_rate=learning_rate, decay=3e-6)

    return optimizer_configured


def create_callbacks(logdir, hparams=None): 
    '''
    '''

    if logdir == None: 
        return None

    callbacks = list()
    tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = logdir,
            update_freq            = 16, 
            profile_batch          = 0
        )
    callbacks.append(tensorboard_callback)
    callbacks.append(hp.KerasCallback(logdir, hparams))

    return callbacks

def create_hyperparams_domains(): 
    '''
    '''
    HP_REGULARIZER = hp.HParam('regularizer', hp.RealInterval(0.1, 0.3))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    HP_KERNEL_NUMBER =  hp.HParam('kernel_number', hp.Discrete([2, 4, 8]))
    HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3, 5]))
    HP_MAX_POOL_SIZE = hp.HParam('max_pool_size', hp.Discrete([2, 4]))

    HP_TRAIN_LOSS = hp.Metric("loss", group="train", display_name="training loss")
    HP_VAL_LOSS   = hp.Metric("val_loss", group="validation", display_name="validation loss")
                                    
    hp_domains = {'kernel_size'           : HP_KERNEL_SIZE,
                  'dropout'               : HP_DROPOUT,
                  'optimizer'             : HP_OPTIMIZER, 
                  'regularizer'           : HP_REGULARIZER, 
                  'kernel_number'         : HP_KERNEL_NUMBER,
                  'max_pool_size'         : HP_MAX_POOL_SIZE,
                  }
    metrics = [HP_TRAIN_LOSS, HP_VAL_LOSS] 

    return hp_domains, metrics
