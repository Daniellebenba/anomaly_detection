from datetime import datetime
from pathlib import Path
import pandas as pd
import json

from .train import train



def ts_split(df):
    '''this function splits and chooses indexes for several train and test sets based on days unit
    and in chronological order. i.e. test set follows the train set by days. 
    Then, the next train-test sets will be the train set before and the new test set will be the one that follows.a
    returns the dictionary cv_splits. 
    For each cv_i we have:
    1. the end index of train set (train set starts always from 0)
    2. start index of test set follows in chronological order the train set
    3. the end index of test set which is: test start index + test_size_days '''
    
    # Create dictionary of cv splits
    test_size_days = 3  # that is the test set window based on days
    test_ind_start = 17  # the first train set will be from size: test_ind_start-1
    test_ind_end = test_ind_start + test_size_days 
    df.ts = pd.to_datetime(df.ts)
    n_days = df.ts.dt.date.nunique()
    arr_days  = df.ts.dt.date.unique()

    cv_splits = {}
    cv_i = 0
    while test_ind_end <= n_days:
        cv_splits[cv_i] = {'train_end': arr_days[test_ind_start-1],
                       'test_start': arr_days[test_ind_start],
                       'test_end': arr_days[test_ind_end-1],} 
        # update indexes
        test_ind_start = test_ind_end 
        test_ind_end = test_ind_start + test_size_days
        cv_i += 1
    return cv_splits


def generate_params(param_grid):
    '''generate random sample from grid search'''
    
    from random import sample
    
    param_res = {}
    for p in param_grid:
        param_res[p] = sample(param_grid[p], 1)[0]
    return param_res

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def cross_validation(
                     params_grid: dict,
                     trainset: pd.DataFrame, 
                     testset: pd.DataFrame,
                     model_config: dict, 
                     output_dir: Path=Path('/home/ubuntu/nabu/anomaly_detection/outputs'),
                     max_iter: int=5,
                     fit_config: dict=None,
                     pipeline=None, 
                     metric: str='auc'):
    
    yyyymmddHHMM = datetime.now().strftime('%Y%m%d%H%M')
    output_path = output_dir / f'{model_config["model_name"]}_results_{yyyymmddHHMM}.csv'
    output_model_path = output_dir / f'{model_config["model_name"]}_model_{yyyymmddHHMM}.pkl'
    output_results = output_dir / f'{model_config["model_name"]}_results_{yyyymmddHHMM}.json'
    
    # split indexes
    cv_splits = ts_split(trainset)    

    cv_dict = {}
    best_params = {}
    
    for grid_i in range(max_iter):
        
        print(f'Parameters Random Search iterate number {grid_i+1}')

        params = generate_params(params_grid) # sample hyperparameters
        params["random_state"] = 100
        print(f'sampled params are: {params}')
        
        cv_dict[grid_i] = {}  # init dictionary for grid_i
        cv_dict[grid_i]['params'] = params    # saves the sampled parameters  
        model_config['params'] = params 

        cv_results = []
        for cv_i in range(len(cv_splits.keys())):
            print(f'CV iterate number {cv_i+1}')
            
            # split to train-val
            df_train_sub = trainset.loc[trainset.ts.dt.date <= cv_splits[cv_i]['train_end']]
            df_val = trainset.loc[(trainset.ts.dt.date >= cv_splits[cv_i]['test_start']) & \
                                 (trainset.ts.dt.date <= cv_splits[cv_i]['test_end'])]
            print(f'train set size: {df_train_sub.shape}, val set size: {df_val.shape}')
            cv_results.append(train(df_train_sub.drop('ts', axis=1), df_val.drop('ts', axis=1), model_config, comment=f'params: {params}', fit_config=fit_config, pipeline=pipeline))
        
        cv_dict[grid_i]['cv_results'] = cv_results
        results = train(trainset.drop('ts', axis=1), testset.drop('ts', axis=1), model_config, comment=f'params: {params}', 
                                fit_config=fit_config, pipeline=pipeline)
        cv_results.append(results)
        
        avg_results = {}
        for res in cv_results:
            for key, val in res.items():
                if key not in ['comment','train_duration', 'time', 'model_name']:
                    avg_results[key] = avg_results.get(key, 0) + (val / len(cv_results))      

        cv_dict[grid_i]['mean_results'] = avg_results
        
        # save best parameters
        if best_params.get(metric, -1) < avg_results.get(metric, -1):
            best_params = params

    # Write report to file
    try:
        json.dump(cv_dict, output_results.open('w'), cls=NpEncoder)
    except:
        print('Error saving CV Results')
    return cv_dict, best_params
