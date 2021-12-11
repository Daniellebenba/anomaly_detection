import json
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from ipaddress import IPv4Address
import datetime

###############
### DATA ######
###############

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
#     df['protocol'] = df.protocol.astype('category')
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


#######################
### PREPROCESSING #####
#######################


def drop_columns(df: pd.DataFrame, cols_names: list) -> pd.DataFrame:
    return df.drop(cols_names, axis=1)


def handle_missing(df: pd.DataFrame, missing: dict) -> pd.DataFrame:
    for col, value in missing.items():
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.add_categories([value])
        df[col] = df[col].fillna(value)
    return df


def create_aggregatoins(df: pd.DataFrame) -> pd.DataFrame:
    df['accum_packet_size'] = df.sort_values(by='ts').groupby('session_id')['packet_size'].cumsum()
    
    df['accum_count_by_ip'] = 1
    df['accum_count_by_ip'] = df.sort_values(by='ts').groupby('ip')['accum_count_by_ip'].cumsum()
    return df


def ipv4_to_dummy_bits(df: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.apply(lambda i: pd.Series([int(b) for b in format(int(IPv4Address(i['ip'])), '032b')], dtype='int8'), axis=1)
    df_tmp.columns = df_tmp.columns.map(lambda bit_idx: f'ip_bit_{bit_idx}')
    return pd.concat([df, df_tmp], axis=1)


# ----------------------------------------------------------------------------------------------------------------------------

def to_numeric_freq_based(df: pd.DataFrame, cols_to_encode: list):
    '''encode to numeric categorical based on occurance frequency'''
    for col in cols_to_encode:       
        sorted_indices = df[col].value_counts().index
        df[col] = df[col].map(dict(zip(sorted_indices, range(1, len(sorted_indices)+1))))
    return df

def ip_encode(x):
    '''add leading zeros and remove dots'''
    return '.'.join(i.zfill(3) for i in x.split('.')).replace('.','')

def time_proc(x):
    return x.dayofweek*24 + x.hour
    
    
def based_paper_preprocess(df: pd.DataFrame, cols_to_encode: list = ['port','country', 'session_id','sensor_id']):
    '''based the paper: https://arxiv.org/pdf/2104.13190.pdf'''
    
    # Timestamp
    df['time_val'] = df.ts.apply(time_proc) 
        
    # ip 
    df.ip = df.ip.apply(ip_encode)
    
    # categorical features
    df = to_numeric_freq_based(df, cols_to_encode)
    
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


###################
### MODELLING #####
###################

######################
### IF/ IF-KMeans ####
######################




###########
### AE ####
###########


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    
   