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
synthetic_dir='/Users/ekargian/Library/Mobile Documents/com~apple~CloudDocs/Documents/g-2/Trackers/ML_tracking/synthetic_data/pkl/'
rnn_dir='/Users/ekargian/Library/Mobile Documents/com~apple~CloudDocs/Documents/g-2/Trackers/ML_tracking/track_finding/RNN'
events_file=open(synthetic_dir+'synthetic_events.pkl','rb')
evts_hits,evts_ids = pickle.load(events_file,encoding='latin1')
if 'synthetic' in synthetic_dir:
    print('\n Reading in %d synthetic events' % evts_hits.shape[0])

    
### Build the NN
tracker_NN=clusterer.Clusterer(detector=tracker,
                               hidden_dim=120,
                               batch_size=100, 
                               n_epochs=12, val_frac=0.1)

tracker_NN.build_model(model_structure='LSTMx2_Dropout')
for seed_location in ['front','middle','back']:
    tracker_NN.model[seed_location].summary()

### Train
'''
from sklearn.model_selection import train_test_split
evts_hits_train, evts_hits_valid, evts_ids_train, evts_ids_valid = train_test_split(evts_hits,evts_ids, train_size=0.8)
tracker_NN.fit(evts_hits_train, evts_ids_train)
util.draw_train_history(tracker_NN.history)
'''
### We can choose to save the model after fitting
'''
for seed_location in ['front','middle','back']:
    tracker_NN.model[seed_location].save(rnn_dir+'/saved_models/tracker_NN_%s_2112_multiseed.h5'%seed_location) 
'''

### Or just read an existing model (must have same model structure)
for seed_location in ['front','middle','back']:
    tracker_NN.model[seed_location] = keras.models.load_model(rnn_dir+'/saved_models/tracker_NN_%s_2112_multiseed.h5'%seed_location)
    tracker_NN.model[seed_location].summary()
