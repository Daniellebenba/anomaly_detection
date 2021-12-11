import pandas as pd
from ipaddress import IPv4Address


def drop_columns(df: pd.DataFrame, cols_names: list) -> pd.DataFrame:
    return df.drop(cols_names, axis=1)


def handle_missing(df: pd.DataFrame, missing: dict) -> pd.DataFrame:
    for col, value in missing.items():
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.add_categories([value])
        df[col] = df[col].fillna(value)
    return df


def create_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregation based on session id 
    grouped = df.sort_values(by='ts').groupby('session_id')
    df['agg_session_id_row_count'] = grouped.cumcount() + 1
    df['agg_session_id_packet_size_sum'] = grouped['packet_size'].cumsum()
    df['agg_session_id_packet_size_max'] = grouped['packet_size'].cummax()
    df['agg_session_id_packet_size_min'] = grouped['packet_size'].cummin()
    df['agg_session_id_packet_size_mean'] = grouped.expanding()['packet_size'].mean().reset_index(level=0, drop=True)
    df['agg_session_id_packet_size_std'] = grouped.expanding()['packet_size'].std().reset_index(level=0, drop=True)

    # Aggregation based on the same ip and the same day
    df['date'] = df['ts'].dt.date
    df['agg_ip_row_count'] = df.sort_values(by='ts').groupby(['ip', 'date']).cumcount() + 1
    df.drop('date', axis=1, inplace=True)

    return df


def ipv4_to_dummy_bits(df: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.apply(lambda i: pd.Series([int(b) for b in format(int(IPv4Address(i['ip'])), '032b')], dtype='int8'),
                      axis=1)
    df_tmp.columns = df_tmp.columns.map(lambda bit_idx: f'ip_bit_{bit_idx}')
    return pd.concat([df, df_tmp], axis=1)


def port_to_dummy_bits(df: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.apply(lambda i: pd.Series([int(b) for b in format(int(i['port']), '016b')], dtype='int8'), axis=1)
    df_tmp.columns = df_tmp.columns.map(lambda bit_idx: f'port_bit_{bit_idx}')
    return pd.concat([df, df_tmp], axis=1)


# ----------------------------------------------------------------------------------------------------------------------------


def mapping_to_numeric_freq_based(df: pd.DataFrame, cols_to_encode: list):
    """create encoding to numeric of categorical variables based on occurance frequency
    returns dictionary with the mapping"""
    mapping_dic = {}
    for col in cols_to_encode:
        mapping_dic[col] = {}
        sorted_indices = df[col].value_counts().index
        mapping_dic[col] = dict(zip(sorted_indices, range(1, len(sorted_indices) + 1)))
        return mapping_dic


def transform_mapping(df: pd.DataFrame, mapping_dic: dict):
    """encode to numeric categorical based on given mapping
    returns the df with cols encoded"""
    cols_to_encode = mapping_dic.keys()
    for col in cols_to_encode:
        df.loc[:, col] = df[col].map(mapping_dic[col])
        return df


def encode_n_most_common(df: pd.DataFrame, col: str, top_labels: list = None, n_top: int = 10) -> pd.DataFrame:
    """if top_labels is given use the list to map.
    else, find N_TOP most common and label others as other group"""

    if top_labels is None:
        top_labels = df[col].value_counts().nlargest(n_top).keys()
    df[f'{col}_most_common_enc'] = np.where(df[col].isin(top_labels), df[col], 'other')
    # Dummy Encoding
    df = pd.concat([df, pd.get_dummies(df[f'{col}_most_common_enc'])], axis=1)
    del df['other'], df[col], df[f'{col}_most_common_enc']
    return df


def ip_encode(x):
    '''add leading zeros and remove dots'''
    return '.'.join(i.zfill(3) for i in x.split('.')).replace('.', '')


def time_proc(x):
    return x.dayofweek * 24 + x.hour


def based_paper_preprocess(df: pd.DataFrame, cols_to_encode: list = ['port', 'country', 'session_id', 'sensor_id']):
    """based the paper: https://arxiv.org/pdf/2104.13190.pdf"""

    # Timestamp
    df.ts = pd.to_datetime(df.ts)
    df['time_val'] = df.ts.apply(time_proc)

    # ip 
    df['ip'] = df.ip.apply(ip_encode)

    return df
