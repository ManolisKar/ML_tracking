import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle

import seeding
import util
import clusterer
import detector
import seed_merging
import keras


import argparse
parser = argparse.ArgumentParser(description=__doc__, epilog=' ', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model-structure',default='LSTMx2_Dropout', help='NN model structure. options include: LSTMx2, LSTMx2_Dropout, LSTMx2_Dropoutx2 (though the latter may crash) ')
parser.add_argument('--hidden-dim',type=int,default=100,help='hidden dimension size')
parser.add_argument('--hidden-dim-2',type=int,default=0,help='hidden dimension size for 2nd layer, if applicable; 0 means same as hidden_dim')
parser.add_argument('--batch-size',type=int,default=100,help='batch size')
parser.add_argument('--epochs',type=int,default=30,help='number of training epochs')
parser.add_argument('--outfile',default=None, help='results output file')
args = parser.parse_args()

model_structure=args.model_structure
hidden_dim=args.hidden_dim
hidden_dim_2=args.hidden_dim_2
if hidden_dim_2==0: hidden_dim_2=hidden_dim
batch_size=args.batch_size
n_epochs=args.epochs
outfile_name=args.outfile


def main():

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
    start_time=time.perf_counter()
    n_epochs_local=n_epochs ## we may modify this, so let's avoid modifying the global -- and avoid the UnboundLocalError
    tracker_NN=clusterer.Clusterer(detector=tracker, 
                                hidden_dim=hidden_dim, hidden_dim_2=hidden_dim_2,
                                batch_size=batch_size, n_epochs=n_epochs_local, val_frac=0.1)
    tracker_NN.build_model(model_structure=model_structure)
    for seed_location in ['front','middle','back']:
        tracker_NN.model[seed_location].summary()
    n_epochs_local=tracker_NN.n_epochs

    ### Train
    from sklearn.model_selection import train_test_split
    evts_hits_train, evts_hits_valid, evts_ids_train, evts_ids_valid = train_test_split(evts_hits,evts_ids, train_size=0.8)
    tracker_NN.fit(evts_hits_train, evts_ids_train)
    util.draw_train_history(tracker_NN.history)
    end_time=time.perf_counter()
    print('Training time (s) : ', end_time-start_time)

    ### We can choose to save the model after fitting
    '''
    for seed_location in ['front','middle','back']:
        tracker_NN.model[seed_location].save(rnn_dir+'/saved_models/tracker_NN_%s_2112_multiseed.h5'%seed_location) 
    '''

    ### Or just read an existing model (must have same model structure)
    '''
    for seed_location in ['front','middle','back']:
        tracker_NN.model[seed_location] = keras.models.load_model(rnn_dir+'/saved_models/tracker_NN_%s_2112_multiseed.h5'%seed_location)
        tracker_NN.model[seed_location].summary()
    '''
    
    ### Get accuracy metrics
    ## read in the 2-track synthetic events for scoring
    events_file=open(synthetic_dir+'synthetic_events.pkl','rb')
    evts_hits_2track,evts_ids_2track = pickle.load(events_file,encoding='latin1')
    evts_hits_test=evts_hits_2track[:1000]
    evts_ids_test=evts_ids_2track[:1000]

    ## First the raw selection score
    start_time=time.perf_counter()
    raw_scores=util.raw_score(tracker_NN,evts_hits_test,evts_ids_test)
    end_time=time.perf_counter()
    print('Time for raw scoring (s) : ', end_time-start_time)

    ## Then score for accuracy/error after seed merging
    start_time=time.perf_counter()
    predicted_ids=seed_merging.predict_events(tracker_NN, evts_hits_test)#,verbose=1)
    full_score=util.score_function(evts_ids_test, predicted_ids )
    end_time=time.perf_counter()
    n_tracks=0
    for predicted_id in predicted_ids:
        n_tracks += np.max(predicted_id)
    print('Total tracks found: ', n_tracks)
    print('Time for full scoring (s) : ', end_time-start_time)

    ## Output results
    if outfile_name is not None:
        print('\nwriting out results to',outfile_name)
        outfile = open(outfile_name,'wb')
        hyperparameters={}
        hyperparameters['model_structure']=model_structure
        hyperparameters['hidden_dim']=hidden_dim
        hyperparameters['hidden_dim_2']=hidden_dim_2
        hyperparameters['batch_size']=batch_size
        hyperparameters['n_epochs']=n_epochs_local
        pickle.dump(
        {
            'hyperparameters':hyperparameters, 
            'raw_scores':raw_scores, 
            'full_score':full_score,
            'n_tracks':n_tracks
        }, 
        outfile
        )
        outfile.close()

    return raw_scores, full_score, n_tracks


if __name__ == "__main__":
    main()

