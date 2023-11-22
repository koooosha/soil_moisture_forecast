import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 



def plot_moist_random_loc(data:pd.DataFrame, random_loc_num:int):
    '''
        This function plots soil moisture of some locations in one plot
        random_loc_num: this variables indicates how many random locations 
        need to be plotted.
    '''
    rand_loc_ids = random.sample(range(1, data['loc'].nunique() + 1),\
        random_loc_num)
    fig = px.line()
    for lc in rand_loc_ids:
        loc_data = data[data["loc"]==lc]
        fig.add_scatter(x=loc_data['time'], y = loc_data['soil_moist'], \
            mode = 'lines', name = f"location: {lc}")
    fig.update_layout(title='Comparison of soil moistures', xaxis_title='Time Stamps', yaxis_title='Soil Moisture')
    fig.show()
    
def plot_dist_moist_random_loc(data:pd.DataFrame):
    '''
        This function plots the distribution of soil moisture for 9 randomly selected locations.
    '''
    number_of_locs = 9
    rand_loc_ids = random.sample(range(1, data['loc'].nunique() + 1), number_of_locs)
    fig, axs = plt.subplots(3,3,figsize=(20,10))
    fig.tight_layout(pad = 5.0)
    loc_idx = 0
    for row in range(3):
        for col in range(3):
            loc_data = data[data["loc"]==rand_loc_ids[loc_idx]]
            loc_data.reset_index(inplace=True)
            sns.histplot(data=loc_data,x='soil_moist', kde=True,ax=axs[row][col])
            axs[row][col].set_title(
                f"soil moist dist at loc id {rand_loc_ids[loc_idx]}")
            loc_idx+=1
    fig.show()
    
def plot_box(data:pd.DataFrame,variables:list):
    """
        This function plots box plots for checking the distribution of a variable and
        checking if any outliers exist or not.
    """
    fig = px.box(data, y= variables, points='all')
    fig.show()
    
def plot_histogram(data:pd.DataFrame,variables:list):
    """
        This function plots histogram listed variables
    """
    fig = px.histogram(data, x= variables)
    fig.show()
    
def plot_training(train_losses,val_losses):
    ''' 
        This function plots the result of the model training.
    '''
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_predictions(predictions,y_test):
    '''
        This methods plot the predicted results (scaled)
    '''
    random_samples = 200
    samples_idx = random.sample(range(1, len(predictions) + 1),random_samples)
    pred_smp = [predictions[idx][0] for idx in samples_idx]
    labels_smp = [y_test[idx] for idx in samples_idx]
    fig = px.line()
    fig.add_scatter(x=list(range(random_samples)), y = pred_smp, \
            mode = 'lines+markers', name = f"Predictions")
    fig.add_scatter(x=list(range(random_samples)), y = labels_smp, \
            mode = 'lines+markers', name = f"Labels")
    fig.update_layout(title='Model Training Eval', xaxis_title='sample_num', yaxis_title='Soil Moisture')
    fig.show()
    return samples_idx
def plot_predictions_unscaled(predictions_unscaled,labels_unscaled,samples_idx):
    '''
        This methods plot the predicted results (scaled)
    '''
    pred_smp_unscaled = [predictions_unscaled[idx] for idx in samples_idx]
    labels_smp_unscaled = [labels_unscaled[idx] for idx in samples_idx]
    fig = px.line()
    fig.add_scatter(x=list(range(len(samples_idx))), y = pred_smp_unscaled, \
            mode = 'lines+markers', name = f"Predictions")
    fig.add_scatter(x=list(range(len(samples_idx))), y = labels_smp_unscaled, \
            mode = 'lines+markers', name = f"Labels")
    fig.update_layout(title='Model Training Eval on Unscaled data', xaxis_title='sample_num', yaxis_title='Soil Moisture')

    fig.show()