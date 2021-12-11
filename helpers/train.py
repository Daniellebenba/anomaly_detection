from models import get_model, get_callbacks
from .eval import eval_model
from helpers import split_x_y, save_model, write_report

from tensorflow.keras import optimizers
from datetime import datetime
from pprint import pprint
from pathlib import Path
import pandas as pd


def train_classic_ml(trainset: pd.DataFrame, 
                     testset: pd.DataFrame, 
                     model_config: dict, 
                     comment: str=None,
                     train_normals_only: bool=True,
                     output_path: Path=None,
                     output_model_path: Path=None):
    print('loading model')
    model = get_model(**model_config)
    
    if train_normals_only: # exclude anomalies
        trainset = trainset.loc[trainset.identify==0]
    X_train, y_train = split_x_y(trainset)
                     
    print('starting training')
    start = datetime.now()
    model.fit(X_train)
    duration = datetime.now() - start
    print(f'train duration took: {duration}')
    if output_model_path:
        save_model(model, output_model_path) 
                     
    X, y = split_x_y(testset)
    print('starting evaluation')
    results = eval_model(X, y, model)
    results['model_name'] = model_config["model_name"]
    results['train_duration'] = str(duration)
    results['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results['comment'] = comment
    print('done. writing results')
    
    pprint(results)
    
    if output_path:
        write_report(results, output_path)
    
    return results


def train_dl(trainset: pd.DataFrame, 
             testset: pd.DataFrame, 
             model_config: dict, 
             comment: str=None, 
             fit_config: dict=None, 
             output_path: Path=None, 
             pipeline=None, 
             validation_size=0.2, 
             output_model=False):   
    trainset = trainset.loc[trainset.identify==0]
    training_sample = round(len(trainset) * validation_size)

    X_train = trainset.iloc[:training_sample]
    X_val = trainset.iloc[training_sample:]
    
    if model_config['params']['optimizer'] == 'adam':
        model_config['params']['optimizer'] = optimizers.Adam(learning_rate=model_config['params']['learning_rate'])
    elif model_config['params']['optimizer'] == 'rmsprop':
        model_config['params']['optimizer'] = optimizers.RMSprop(learning_rate=model_config['params']['learning_rate'])
    
    y_train = X_train.pop('identify')
    y_val = X_val.pop('identify')
    y_test = testset.pop('identify')
        
    if pipeline:
        print('transform datasets')
        pipeline.fit(trainset.drop('identify', axis=1))
        X_train = pipeline.transform(X_train)  # check if remove anomalies
        X_val = pipeline.transform(X_val)
        X_test = pipeline.transform(testset)
    else:
        X_test = testset
    
    print('loading model')
    model = get_model(**model_config)
    cb = get_callbacks(model)
    
    print('starting training')   
    start = datetime.now()
    history = model.fit(
        X_train, X_train,
        shuffle=True,
        epochs=fit_config['epochs'],
        batch_size=fit_config['batch_size'],
        callbacks=cb,
        validation_data=(X_val, X_val)
    )
    duration = datetime.now() - start
    print(f'train duration took: {duration}')
    
    results = eval_model(X_test, y_test, model, is_dl=True)
    results['model_name'] = model_config['model_name']
    results['train_duration'] = duration
    results['time'] = str(datetime.now())
    results['comment'] = comment
                     
    print('done. writing results')
    if output_path:
        write_report(results, output_path)
        
    return results, model if output_model else results

def train(trainset, testset, model_config, comment, fit_config, pipeline=None):
    if model_config['model_name'] in ('IF', 'IF_KMeans'):
        return train_classic_ml(trainset,
                                testset, 
                                model_config, 
                                comment)
    elif model_config['model_name'] in ('AutoEncoder'):
        return train_dl(trainset, 
                        testset, 
                        model_config, 
                        comment, 
                        fit_config, 
                        pipeline=pipeline)
    else: 
        raise ValueError(f"{model_config['model_name']} not supported") 