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

load_saved_model=1

### Build the NN
tracker_NN=clusterer.Clusterer(hidden_dim_1=250,
                               hidden_dim_2=250,
                               dense_dim=400,
                               nstraws_perlayer=4,
                               batch_size=50, 
                               n_epochs=4000, patience=10,
                               val_frac=0.15, detector=tracker)
tracker_NN.build_model(model_structure='LSTMx2_Dropout_ExtraDense',
                       loss='categorical_crossentropy')
for seed_location in ['front','middle','back']:
    if load_saved_model:
        tracker_NN.model[seed_location] = keras.models.load_model(rnn_dir+'/saved_models/tracker_NN_2202_noise_4StrawsPerLayer_%s.h5'%seed_location)
    tracker_NN.model[seed_location].summary()

### Train
from sklearn.model_selection import train_test_split
#evts_hits_train, evts_hits_test, evts_ids_train, evts_ids_test = train_test_split(evts_hits,evts_ids, train_size=1-val_frac)
#tracker_NN.fit(evts_hits_train, evts_ids_train)
tracker_NN.fit(evts_hits, evts_ids)
#util.draw_train_history(tracker_NN.history)

### We can choose to save the model after fitting
for seed_location in ['front','middle','back']:
    tracker_NN.model[seed_location].save(rnn_dir+'/saved_models/tracker_NN_2203_noise_4StrawsPerLayer_%s.h5'%seed_location) 

