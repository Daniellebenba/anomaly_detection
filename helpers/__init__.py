import csv
import pickle
import pandas as pd
from pathlib import Path


def set_seed(seed: int):
    import random
    import numpy as np
    # import torch
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    tf.random.set_seed(seed)    


def write_report(results: dict, output_path: Path):
    is_exists =  output_path.exists()
    with output_path.open('a', newline='') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not is_exists:
            writer.writeheader()
        writer.writerow(results)
        
        
def save_model(model, output_path: Path):
    with output_path.open('wb') as f:
        pickle.dump(model, f)
        

def split_x_y(df: pd.DataFrame):
    # X-y sets
    to_drop = ['identify']  
    X = df.drop(to_drop, axis=1)
    y = df.identify    
    return X, y


from .eval import eval_model, use_mad_threshold
from .metrics import get_aucpr, get_auc, root_mean_squared_error
from .train import train_classic_ml, train_dl, train
from .cross_validation import ts_split, generate_params, cross_validation
from .plots import (
    plot_metric,
    figure,
    plot_prediction_curves,
    plot_confusion_matrix
)

  
__all__ = [
    'set_seed',
    'get_aucpr', 
    'get_auc',
    'root_mean_squared_error',
    'plot_metric',
    'figure',
    'plot_prediction_curves',
    'plot_confusion_matrix',
    'write_report',
    'save_model',
    'split_x_y',
    'eval_model',
    'train_dl',
    'train_classic_ml',
    'train',
    'ts_split', 
    'generate_params', 
    'cross_validation',
    'use_mad_threshold'
    ]
