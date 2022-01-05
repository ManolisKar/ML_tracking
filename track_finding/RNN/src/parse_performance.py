import os,sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle
import glob

## Hold trial results in this dict
results={}
hyperparameters=['model_structure','activation_func','optimizer','hidden_dim','hidden_dim_2','dropout_rate','batch_size','n_epochs']
for hyperparameter in hyperparameters: results[hyperparameter]=[]
metrics=['raw_score_avg','raw_error_avg','full_score','full_error', 'n_tracks']
for metric in metrics: results[metric]=[]

## Read in trial results
ml_tracking_dir=os.environ['ML_TRACKING']
performance_data_dir=ml_tracking_dir + '/track_finding/RNN/optimization_data/'
pkl_filenames=glob.glob(performance_data_dir + 'performance_*.pkl')
## Read through all files, adding the results to the dict lists
for pkl_filename in pkl_filenames:
    print('Reading file ', pkl_filename)
    this_trial={}
    performance_trial = pd.read_pickle(pkl_filename)
    # first add the hyperparameters
    for hyperparameter in hyperparameters:
        this_trial[hyperparameter] = performance_trial['hyperparameters'][hyperparameter]
    # next get the raw scores, and average over the detector regions
    raw_score_avg, raw_error_avg =0,0
    for region in ['front','middle','back']:
        raw_score_avg += performance_trial['raw_scores'][region][0]
        raw_error_avg += performance_trial['raw_scores'][region][1]
    if np.isnan(raw_score_avg) or np.isnan(raw_error_avg): continue
    if performance_trial['full_score']==0: continue
    raw_score_avg/=3
    raw_error_avg/=3
    this_trial['raw_score_avg']=raw_score_avg
    this_trial['raw_error_avg']=raw_error_avg
    this_trial['full_score']=performance_trial['full_score'][0]
    this_trial['full_error']=performance_trial['full_score'][1]
    this_trial['n_tracks']=performance_trial['n_tracks']
    # if no nan's or errors, add results from this trial
    for hyperparameter in hyperparameters: results[hyperparameter]+= [this_trial[hyperparameter]]
    for metric in metrics: results[metric]+=[this_trial[metric]]


