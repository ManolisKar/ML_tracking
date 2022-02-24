import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

import seeding
import util
import clusterer
import detector
import seed_merging
import keras

tracker=detector.build_tracker()

## Read in synthetic events
import pickle
ml_tracking_dir=os.environ['ML_TRACKING']
synthetic_dir=ml_tracking_dir+'/synthetic_data/pkl/'
rnn_dir=ml_tracking_dir + '/track_finding/RNN/'
## read in the 2-track synthetic events for scoring
events_file=open(synthetic_dir+'tracked_events_noise_WithSingles.pkl','rb')
evts_hits,evts_ids,tracked_ids = pickle.load(events_file,encoding='latin1')
if 'synthetic' in synthetic_dir: print('\n Reading in %d synthetic events' % evts_hits.shape[0])

### Build the NN
tracker_NN=clusterer.Clusterer(hidden_dim_1=250,
                               hidden_dim_2=250,
                               dense_dim=400,
                               batch_size=100, 
                               n_epochs=400, val_frac=0.05, detector=tracker)
tracker_NN.build_model(model_structure='LSTMx2_Dropout_ExtraDense')
for seed_location in ['front','middle','back']:
    tracker_NN.model[seed_location].summary()

### Train
from sklearn.model_selection import train_test_split
evts_hits_train, evts_hits_test, evts_ids_train, evts_ids_test = train_test_split(evts_hits,evts_ids, train_size=0.95)
tracker_NN.fit(evts_hits_train, evts_ids_train)
util.draw_train_history(tracker_NN.history)

### We can choose to save the model after fitting
for seed_location in ['front','middle','back']:
    tracker_NN.model[seed_location].save(rnn_dir+'/saved_models/tracker_NN_%s_2202_noise.h5'%seed_location) 

### Or just read an existing model (must have same model structure)
'''
for seed_location in ['front','middle','back']:
    tracker_NN.model[seed_location] = keras.models.load_model(rnn_dir+'/saved_models/tracker_NN_%s_2112_multiseed.h5'%seed_location)
    tracker_NN.model[seed_location].summary()
'''
