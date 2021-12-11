############# ALGORITHM CLASSES ##############


# We implement the following algorithms classes:
# 1. IF : we use sklearn implementation
# 2. IF-Kmeans : based on the paper: https://arxiv.org/pdf/2104.13190.pdf
# 3. Random Cut Forest : https://github.com/kLabUM/rrcf
# 4. AE with Entity Embeddings : based on the paper: https://arxiv.org/pdf/1910.02203.pdf

##### Isolation Forest #####

# use sklearn
from sklearn.ensemble import IsolationForest

##### IForest-KMeans #####
from .IF_KMeans import IF_KMeans

##### AE - Entity Embeddings ##### 
from .AutoEncoders import AutoEncoders

from tensorflow.keras import callbacks
from datetime import datetime


def get_model(model_name: str, params: dict, input_dim: int = 0):
    if model_name == 'IF':
        return IsolationForest(**params)
    elif model_name == 'IF_KMeans':
        return IF_KMeans(**params)
    elif model_name == 'AutoEncoder':
        if not input_dim:
            print('Error: you must input input_dim size')
            return None
        model = AutoEncoders(input_dim, params['architecture'])
        model.compile(optimizer=params['optimizer'],
                      loss=params['loss'],
                      metrics=params['metrics'])
        return model


def get_callbacks(model, batch_size: int = 256):
    yyyymmddHHMM = datetime.now().strftime('%Y%m%d%H%M')

    log_subdir = f'{yyyymmddHHMM}_batch{batch_size}_layers{len(model.layers)}'

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    save_model = callbacks.ModelCheckpoint(
        filepath=f'{model.name}_best_weights.tf',
        save_best_only=True,
        monitor='val_loss',
        verbose=0,
        mode='min'
    )

    return [early_stop, save_model]


__all__ = [
    'IsolationForest',
    'IF_KMeans',
    'AutoEncoders',
    'get_model'
]
