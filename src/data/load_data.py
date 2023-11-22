import pandas as pd

def load_data_file(path:str)->pd.DataFrame:
    '''This function loads data set from the input path'''
    return pd.read_csv(path)