import json
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import datetime


def load_data(file_path: Path) -> pd.DataFrame:
    global_sh_file = json.load(file_path.open('r'))

    global_sh_data = []
    for doc in tqdm(global_sh_file, desc='parsing raws'):
        doc_copy = doc.copy()
        doc_copy['_id'] = doc['_id']['$oid'] if isinstance(doc['_id'], dict) else doc['_id']
        if 'raw_data' in doc:
            doc_copy['raw_data'] = doc['raw_data']['$binary']['base64'] if isinstance(doc['raw_data'], dict) else doc['raw_data']
        doc_copy['ts'] = doc['ts']['$date'] if isinstance(doc['ts'], dict) else doc['ts']
        global_sh_data.append(doc_copy)
        
    df = pd.DataFrame(global_sh_data)
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    before = df.memory_usage(index=False, deep=True).sum()
    # reduce memory usage
    df['country'] = df.country.astype('category')
    df['sensor_id'] = df.sensor_id.astype('category')
    df['ip'] = df.ip.astype('category')
    df['ts'] = pd.to_datetime(df.ts)
    after = df.memory_usage(index=False, deep=True).sum()
    print(f'memory usage reduced from {before} to {after}')
    return df


def add_labels(df: pd.DataFrame, malicious_ids: list) -> pd.DataFrame:
    df['identify'] = False
    df.loc[df['_id'].isin(malicious_ids), 'identify'] = True
    return df


def split_by_date(df: pd.DataFrame, split_date: datetime.datetime = pd.to_datetime('2021-07-31')):
    
    df_val = df.loc[pd.to_datetime(df.ts.dt.date) >= split_date]
    df_train = df.loc[pd.to_datetime(df.ts.dt.date) < split_date]

    n_train = len(df_train)
    n_val = len(df_val)
    n_total = len(df)

    print('Splitting data to Train set with {:,} rows and Validation set with {:,} rows. \
     \nWith {}% - {}% ratio'.format(n_train, n_val, round(n_train/n_total*100, 2), round(n_val/n_total*100, 2)))
    
    return df_train, df_val
