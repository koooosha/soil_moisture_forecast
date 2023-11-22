import pandas as pd

def series_seq(data:pd.DataFrame, window_size:int,feat_list:list,label:str="soil_moist"):
    '''
    This function gets a data set and creates sequential data set for lstm model training.
    '''
    feats_data, labels_data = [], []

    for _, data_loc in data.groupby('loc'):
        features = data_loc[feat_list].drop(label,axis=1).values
        labels = data_loc[label].values
        # Creating sqeuntial data set by considering window size
        for idx in range(len(features)-window_size):
            feats_data.append(features[idx:idx+window_size])
            labels_data.append(labels[idx+window_size])
            
    return feats_data,labels_data