import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_location_id(data:pd.DataFrame)->pd.DataFrame:
    '''This function creates a new feature as location id by checking lat and long and 
    find the unique locations and assign an identifier to them.'''
    unique_df = data.drop_duplicates(subset=["latitude","longitude"])
    unique_df.reset_index(inplace=True)
    data['loc'] = np.nan
    for idx, row in unique_df.iterrows():
        data.loc[(data["latitude"]==row["latitude"]) & \
            (data["longitude"] == row["longitude"]),'loc'] = idx+1 
    return data

def feature_scaling(data:pd.DataFrame, feat_list:list):
    '''
     This function makes a min max scaler and transform numerical feats by it and returns the scaler.
    '''
    mms = MinMaxScaler()
    data[feat_list] = mms.fit_transform(data[feat_list])
    return (mms, data)

def scaling_inverse(data:pd.DataFrame, scaler,numeric_feat:list):
    """
    This function gets back the scaled value to its normal value.
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    scaled_data = np.zeros((len(data), len(numeric_feat)))
    scaled_data[:, -1] = data.ravel()
    unscaled_data = scaler.inverse_transform(scaled_data)
    return unscaled_data[:, -1]