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
parser.add_argument('--modelname',default='test',help='Model files expected in format [modelname]_<seed-location>.h5 in the saved_models dir.')
parser.add_argument('--datafile',default=None,help='Dataset file to score on. Use default if None. pkl file expected.')
args = parser.parse_args()
modelname=args.modelname
datafile=args.datafile

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
if datafile is None:
    ml_tracking_dir=os.environ['ML_TRACKING']
    synthetic_dir=ml_tracking_dir+'/synthetic_data/pkl/'
    ## read in the 2-track synthetic events for scoring
    events_file=open(synthetic_dir+'synthetic_events.pkl','rb')
else: 
    events_file=open(datafile,'rb')
## did we read a file with tracked data?
tracked=0
if 'tracked' in datafile:
    tracked=1
    evts_hits,evts_ids,tracked_ids = pickle.load(events_file,encoding='latin1')
else:
    evts_hits,evts_ids = pickle.load(events_file,encoding='latin1')

ntest=2000
evts_hits_test=evts_hits[:ntest]
evts_ids_test=evts_ids[:ntest]
if tracked: tracked_ids=tracked_ids[:ntest]

### Get accuracy metrics
## First the raw selection score
start_time=time.perf_counter()
raw_scores=util.raw_score(tracker_NN,evts_hits_test,evts_ids_test)
end_time=time.perf_counter()
raw_score_time=end_time-start_time
print('Time for raw scoring (s) : ', raw_score_time)

## Then score for accuracy/error after seed merging
start_time=time.perf_counter()
predicted_ids=seed_merging.predict_events(tracker_NN, evts_hits_test)#,verbose=1)
end_time=time.perf_counter()
predict_time=end_time-start_time
start_time=time.perf_counter()
full_score=util.score_function(evts_ids_test, predicted_ids )
end_time=time.perf_counter()
full_score_time=end_time-start_time
n_tracks=0
for predicted_id in predicted_ids:
    n_tracks += np.max(predicted_id)
print('Total tracks found (alt extraction): ', n_tracks)
print('Time for prediction (s) : ', predict_time)
print('Time for full scoring (s) : ', full_score_time)

## If we have tracked data, give these metrics also
if tracked:
    print('\n\n Metrics for tracked data:')
    full_score=util.score_function(evts_ids_test, tracked_ids )
    n_tracks=0
    for tracked_id in tracked_ids:
        n_tracks += np.max(tracked_id)
    print('Total tracks found (alt extraction): ', n_tracks)
