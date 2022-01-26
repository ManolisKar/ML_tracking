import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle
import random

import seeding
import util
import clusterer
import detector
import seed_merging
import keras


import argparse
parser = argparse.ArgumentParser(description=__doc__, epilog=' ', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--modelname',default='test',help='model files expected in format [modelname]_<seed-location>.h5 in the saved_models dir')
args = parser.parse_args()
modelname=args.modelname

## Build the NN
tracker=detector.build_tracker()
tracker_NN=clusterer.Clusterer(detector=tracker)
tracker_NN.build_model()

## Read in the trained model
for seed_location in ['front','middle','back']:
    tracker_NN.model[seed_location] = keras.models.load_model('saved_models/%s_%s.h5'%(modelname,seed_location))
    tracker_NN.model[seed_location].summary()

## Read in synthetic events
import pickle
ml_tracking_dir=os.environ['ML_TRACKING']
synthetic_dir=ml_tracking_dir+'/synthetic_data/pkl/'
rnn_dir=ml_tracking_dir+'/track_finding/RNN'
events_file=open(synthetic_dir+'synthetic_events.pkl','rb')
evts_hits,evts_ids = pickle.load(events_file,encoding='latin1')
if 'synthetic' in synthetic_dir:
    print('\n Reading in %d synthetic events' % evts_hits.shape[0])
## read in the 2-track synthetic events for scoring
events_file=open(synthetic_dir+'synthetic_events.pkl','rb')
evts_hits_2track,evts_ids_2track = pickle.load(events_file,encoding='latin1')
evts_hits_test=evts_hits_2track[:1000]
evts_ids_test=evts_ids_2track[:1000]
    
'''### Get accuracy metrics
## First the raw selection score
start_time=time.perf_counter()
raw_scores=util.raw_score(tracker_NN,evts_hits_test,evts_ids_test)
end_time=time.perf_counter()
raw_score_time=end_time-start_time
print('Time for raw scoring (s) : ', raw_score_time)
'''

## Then score for accuracy/error after seed merging
start_time=time.perf_counter()
predicted_ids=seed_merging.predict_events(tracker_NN, evts_hits_test)#,verbose=1)
full_score=util.score_function(evts_ids_test, predicted_ids )
end_time=time.perf_counter()
full_score_time=end_time-start_time
n_tracks=0
for predicted_id in predicted_ids:
    n_tracks += np.max(predicted_id)
print('Total tracks found: ', n_tracks)
print('Time for full scoring (s) : ', full_score_time)
