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
parser.add_argument('--N',type=int,default=2,help='number of iterations')
parser.add_argument('--outdir',default='optimization_data',help='output directory for data from each iteration')
args = parser.parse_args()
N=args.N
outdir=args.outdir


def get_accuracy_scores(model_structure='LSTMx2_Dropout', 
                        hidden_dim=100, hidden_dim_2=0, dense_dim=100,
                        activation_func='softmax', dense_activation_func='relu',
                        optimizer='Nadam',
                        dropout_rate=0.2,
                        batch_size=100, n_epochs=50, 
                        outfile_name=None):

    ### Build the NN
    start_time=time.perf_counter()
    tracker=detector.build_tracker()
    if hidden_dim_2==0: hidden_dim_2=hidden_dim
    n_epochs_local=n_epochs ## we may modify this, so let's avoid modifying the global -- and avoid the UnboundLocalError
    tracker_NN=clusterer.Clusterer(detector=tracker, 
                                hidden_dim=hidden_dim, hidden_dim_2=hidden_dim_2, dense_dim=dense_dim, dropout_rate=dropout_rate,
                                batch_size=batch_size, n_epochs=n_epochs_local, val_frac=0.1)
    tracker_NN.build_model(model_structure=model_structure, activation_func=activation_func, 
                            dense_activation_func=dense_activation_func, optimizer=optimizer)

    ### Train
    from sklearn.model_selection import train_test_split
    ##### **** just for testing of callback -- remove the :500
    evts_hits_train, evts_hits_valid, evts_ids_train, evts_ids_valid = train_test_split(evts_hits,evts_ids, train_size=0.8)
    tracker_NN.fit(evts_hits_train, evts_ids_train)
    n_epochs_local=tracker_NN.n_epochs
    #util.draw_train_history(tracker_NN.history)
    end_time=time.perf_counter()
    train_time=end_time-start_time
    print('Training time (s) : ', train_time)

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
    ## First the raw selection score
    start_time=time.perf_counter()
    raw_scores=util.raw_score(tracker_NN,evts_hits_test,evts_ids_test)
    end_time=time.perf_counter()
    raw_score_time=end_time-start_time
    print('Time for raw scoring (s) : ', raw_score_time)

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

    ## Output results
    if outfile_name is not None:
        print('\nwriting out results to',outfile_name)
        outfile = open(outfile_name,'wb')
        hyperparameters={}
        hyperparameters['model_structure']=model_structure
        hyperparameters['dense_activation_func']=dense_activation_func
        hyperparameters['optimizer']=optimizer
        hyperparameters['hidden_dim']=hidden_dim
        hyperparameters['hidden_dim_2']=hidden_dim_2
        hyperparameters['dense_dim']=dense_dim
        hyperparameters['dropout_rate']=dropout_rate
        hyperparameters['batch_size']=batch_size
        hyperparameters['n_epochs']=n_epochs_local
        pickle.dump(
        {
            'hyperparameters':hyperparameters, 
            'raw_scores':raw_scores, 
            'full_score':full_score,
            'n_tracks':n_tracks,
            'train_time':train_time,
            'raw_score_time':raw_score_time,
            'full_score_time':full_score_time
        }, 
        outfile
        )
        outfile.close()

    return raw_scores, full_score, n_tracks


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

'''
Iterate with randomly different hyper-parameter choices.
Get accuracy and performance scores in each iteration.
'''
model_structures=['LSTMx2','LSTMx2_Dropout','LSTMx2_ExtraDense','LSTMx2_Dropout_ExtraDense','LSTM_ExtraDense','LSTM_Dropout_ExtraDense']#,'LSTM','LSTMx2_Dropoutx2]
dense_activation_funcs=['relu','sigmoid','softplus','tanh']
optimizers=['Nadam','Adam','RMSprop','Adagrad']#,'SGD']

for i in range(N):
    model_structure=random.choice(model_structures)
    dense_activation_func=random.choice(dense_activation_funcs)
    optimizer=random.choice(optimizers)
    hidden_dim=random.randint(20,300)
    if 'LSTMx2' in model_structure: hidden_dim_2=random.randint(20,300)
    else: hidden_dim_2=0
    if 'Dropout' in model_structure: dropout_rate=random.uniform(0.01,0.4)
    else: dropout_rate=0
    if 'ExtraDense' in model_structure: dense_dim=random.randint(20,1000)
    else: dense_dim=0
    batch_size=random.randint(5,400)
    outfile_name=outdir+'/performance_%d.pkl'%i

    print('model structure: ', model_structure)
    print('dense activation func: ', dense_activation_func)
    print('optimizer: ', optimizer)
    print('hidden_dim: ', hidden_dim)
    print('hidden_dim_2: ', hidden_dim_2)
    print('dense_dim: ', dense_dim)
    print('dropout_rate: ', dropout_rate)
    print('batch_size: ', batch_size)
    try:
        get_accuracy_scores(model_structure=model_structure, 
                hidden_dim=hidden_dim, hidden_dim_2=hidden_dim_2, dense_dim=dense_dim, 
                dense_activation_func=dense_activation_func, optimizer=optimizer,
                batch_size=batch_size, dropout_rate=dropout_rate, n_epochs=2,
                outfile_name=outfile_name)
    except ValueError as err:
        print(err.args)


