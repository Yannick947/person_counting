import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import keras 

def get_optimizer(optimizer, learning_rate=1e-4):
    '''
    '''
    #TODO: For finetuning, more parameters (e.g momentum params) could be tuned like beta1 and beta2 and decay

    if optimizer == 'RMSProp': 
        optimizer_configured = keras.optimizers.RMSprop(learning_rate=learning_rate, decay=learning_rate / 100)

    elif optimizer == 'SGD': 
        optimizer_configured = keras.optimizers.SGD(learning_rate=learning_rate / 10, decay=learning_rate / 100)

    else: 
        optimizer_configured = keras.optimizers.Adam(learning_rate=learning_rate, decay=learning_rate / 100)

    return optimizer_configured


def get_static_hparams(args): 
    '''
    
    '''
    logging_ret = dict()
    LOGGING_ARGS = [
                    'filter_cols_upper',
                    'batch_size',
                    'filter_cols_factor',
                    'filter_rows_factor',
                    'filter_cols_lower'
                    ]

    for key in LOGGING_ARGS:
        if vars(args)[key] is not None: 
            logging_ret[key] = vars(args)[key]

    return logging_ret


def create_callbacks(logdir, hparams=None, save_best=False, reduce_on_plateau=True): 
    '''
    '''

    if logdir == None: 
        return None

    callbacks = list()
    tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = logdir,
            update_freq            = 128, 
            profile_batch          = 0, 
            write_graph            = True,
            write_grads            = False
        )
        
    callbacks.append(tensorboard_callback)
    callbacks.append(hp.KerasCallback(logdir, hparams))

    if save_best: 
        callbacks.append(keras.callbacks.ModelCheckpoint('best_model_{epoch:02d}_{val_loss:.2f}.hdf5',
                                                          monitor='val_loss',
                                                          save_best_only=True, 
                                                          mode='min'))

    if reduce_on_plateau: 
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor    = 'val_loss',
            factor     = 0.05,
            patience   = 3,
            verbose    = 1,
            mode       = 'auto',
            min_delta  = 0.001,
            cooldown   = 0,
            min_lr     = 1e-8
        ))

    return callbacks



