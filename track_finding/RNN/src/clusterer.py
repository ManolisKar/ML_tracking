import numpy as np
import seeding
import keras
import pandas as pd

from sklearn.base import BaseEstimator
from statistics import mode


class Clusterer(BaseEstimator):
    def __init__(self, detector,
                 hidden_dim=50, hidden_dim_2=50, dense_dim=100,
                 dropout_rate=0.2, batch_size=128, n_epochs=5, 
                 val_frac=0.2):
        """
        LSTM model example.
        TODO: fill in more details.
        """
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim_2
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.val_frac = val_frac
        self.detector=detector
        self.nstraws_perlayer = 6 # max number of straw hits per layer (including "0", ie no hits from the track in this layer)
        
        if False: '''
        self.model=None
        self.train_input = None
        self.train_target = None
        self.prepared = False     
        
        Instead, prepare 3 models, one for each region of the detector where a seed is located
        -- front, middle, back.
        '''
        
        self.model={}
        self.train_input = {}
        self.train_target = {}
        self.prepared = {}    
        self.history = {}    
        
        

    def build_model(self, 
                    model_structure='LSTMx2_Dropout',
                    loss='binary_crossentropy',## the HEPTrkX code was using categorical_ce
                    activation_func='softmax', dense_activation_func='relu',
                    optimizer='Nadam', metrics=['accuracy']):
        n_hidden=self.hidden_dim
        n_hidden_2=self.hidden_dim_2
        n_dense=self.dense_dim
        dropout_rate=self.dropout_rate
        length=self.detector.n_layers
        width=self.nstraws_perlayer
        ## flatten the hits/seeds for each straw into a single dimension of 2*width elements
        inputs = keras.layers.Input(shape=(length, 2*width))
        hidden_1, dropout_1, hidden_2, extra_dense, outputs = {},{},{},{},{}
        for seed_location in ['front','middle','back']:

            ##bi-directional LSTM layer:
            hidden_1[seed_location] = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden, return_sequences=True, activation='softmax'))(inputs)
            ## could try adding more LSTM layers, with dropout inbetween to prevent overfitting
            if ('LSTMx2' in model_structure) and ('Dropoutx2' in model_structure):
                dropout_1[seed_location] = keras.layers.Dropout(dropout_rate)(hidden_1[seed_location])
                hidden_2[seed_location] = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden_2, return_sequences=True))(dropout_1[seed_location])
                ## for some very weird reason, adding the 2nd dropout layer crashes on my machine...
                dropout_2[seed_location] = keras.layers.Dropout(dropout_rate)(hidden_2)
                outputs[seed_location] = keras.layers.TimeDistributed( keras.layers.Dense(width+1, activation=activation_func))(dropout_2[seed_location])
            elif ('LSTMx2' in model_structure) and ('Dropout' in model_structure) and ('ExtraDense' in model_structure):
                dropout_1[seed_location] = keras.layers.Dropout(dropout_rate)(hidden_1[seed_location])
                hidden_2[seed_location] = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden_2, return_sequences=True))(dropout_1[seed_location])
                extra_dense[seed_location] = keras.layers.TimeDistributed( keras.layers.Dense(n_dense, activation=dense_activation_func))(hidden_2[seed_location])
                outputs[seed_location] = keras.layers.Dense(width+1, activation='softmax')(extra_dense[seed_location])
            elif ('LSTM' in model_structure) and ('Dropout' in model_structure) and ('ExtraDense' in model_structure):
                dropout_1[seed_location] = keras.layers.Dropout(dropout_rate)(hidden_1[seed_location])
                extra_dense[seed_location] = keras.layers.TimeDistributed( keras.layers.Dense(n_dense, activation=dense_activation_func))(dropout_1[seed_location])
                outputs[seed_location] = keras.layers.Dense(width+1, activation='softmax')(extra_dense[seed_location])
            elif ('LSTMx2' in model_structure) and ('Dropout' not in model_structure) and ('ExtraDense' in model_structure):
                hidden_2[seed_location] = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden_2, return_sequences=True))(hidden_1[seed_location])
                extra_dense[seed_location] = keras.layers.TimeDistributed( keras.layers.Dense(n_dense, activation=dense_activation_func))(hidden_2[seed_location])
                outputs[seed_location] = keras.layers.Dense(width+1, activation='softmax')(extra_dense[seed_location])
            elif ('LSTM' in model_structure) and ('Dropout' not in model_structure) and ('ExtraDense' in model_structure):
                extra_dense[seed_location] = keras.layers.TimeDistributed( keras.layers.Dense(n_dense, activation=dense_activation_func))(hidden_1[seed_location])
                outputs[seed_location] = keras.layers.Dense(width+1, activation='softmax')(extra_dense[seed_location])
            elif ('LSTMx2' in model_structure) and ('Dropout' in model_structure):
                dropout_1[seed_location] = keras.layers.Dropout(dropout_rate)(hidden_1[seed_location])
                hidden_2[seed_location] = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden_2, return_sequences=True))(dropout_1[seed_location])
                #dropout_2[seed_location] = keras.layers.Dropout(dropout_rate)(hidden_2)
                outputs[seed_location] = keras.layers.TimeDistributed( keras.layers.Dense(width+1, activation=activation_func))(hidden_2[seed_location])
            elif ('LSTMx2' in model_structure) and ('Dropout' not in model_structure):
                hidden_2[seed_location] = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden_2, return_sequences=True))(hidden_1[seed_location])
                outputs[seed_location] = keras.layers.TimeDistributed( keras.layers.Dense(width+1, activation=activation_func))(hidden_2[seed_location])
            else:
                outputs[seed_location] = keras.layers.TimeDistributed( keras.layers.Dense(width+1, activation=activation_func))(hidden_1[seed_location])

            self.model[seed_location] = keras.Model(inputs, outputs[seed_location])
            self.model[seed_location].compile(loss=loss, optimizer=optimizer, metrics=metrics)            
          
    '''
    Can we make loss function give bigger weight to events with many hits?
    That would train the model to deal with complexity.
    
    Also consider "momentum acceleration" or other modifications of learning rate for better convergence.
    '''

        
    def prepare_training_data(self, evts_hits, evts_ids):
        """
        Prepare training data. 
        Reads in 3D arrays of hits and ids in many event windows.
        The 1st dimension is the event number.
        The hits and ids within each window  are of the same shape: (length,width).
        Within each window generate seeds.
        *** In this iteration, seeds are genrated anywhere in the tracker, and multiple per track ***
        The input will be of shape (n_seeds, length, width)
        """
        #if self.prepared: return
        #self.prepared = True
        
        n_events = evts_hits.shape[0]
        max_seeds = n_events * self.detector.max_tracks * self.detector.n_modules / 2 # divide by 2 since seeds are split in ~3
        print_freq = int(max_seeds / 20)

        holder_input, holder_target={},{}
        n_seeds={} # this will be a counter for seeds we find
        for seed_location in ['front','middle','back']:
            holder_input[seed_location] = np.zeros((int(max_seeds), 
                                                    self.detector.n_layers, 
                                                    2*self.nstraws_perlayer))
            holder_target[seed_location] = np.zeros((int(max_seeds), 
                                                     self.detector.n_layers, 
                                                     self.nstraws_perlayer+1))
            n_seeds[seed_location]=0
    
        ## Iterate over all windows (events)
        for i_evt in range(n_events):

            ## All hits coordinates in this event
            allhits_layers,allhits_straws=np.where(evts_hits[i_evt]>0)

            ## Get all seeds in this event
            seeds_xy, seeds_array=seeding.make_real_seeds(evts_ids[i_evt])
            if len(seeds_xy)==0: continue  
            
            ## Step through each module
            for i_module in range(self.detector.n_modules):

                if i_module<=2: seed_location='front'
                elif i_module<=4: seed_location='middle'
                else: seed_location='back' 
                
                ## Get seeds in this module
                module_seeds_xy = [seed_xy for seed_xy in seeds_xy if (
                    (seed_xy[0][0]>=i_module*4) & (seed_xy[0][0]<i_module*4+4) )]
                if len(module_seeds_xy)==0: continue

                for seed_xy in module_seeds_xy:
                    ## format input for this seed     
                    
                    # array for hits per layer:
                    # this will randomize where hit coordinates appear in the NN input, to avoid bias
                    ihit_perlayer=np.zeros(shape=(self.detector.n_layers,self.nstraws_perlayer)).astype(int)
                                    
                    ## first turn on the seed hits after finding their (layer,straw) location
                    seed_layers,seed_straws = seed_xy
                    for i,ilayer in enumerate(seed_layers):
                        istraw=seed_straws[i]
                        ## get the randomized input index for this hit
                        ihit_index = np.random.choice( np.array( np.where(ihit_perlayer[ilayer]==0)[0] ))
                        ## turn on both hits and seeds elements
                        holder_input[seed_location][n_seeds[seed_location]][ilayer][ihit_index] = istraw+1
                        holder_input[seed_location][n_seeds[seed_location]][ilayer][self.nstraws_perlayer+ihit_index] = 1
                        ## turn on that straw location at the output
                        holder_target[seed_location][n_seeds[seed_location]][ilayer][ihit_index+1] = 1
                        ## increment i_hit in this layer
                        ihit_perlayer[ilayer][ihit_index]=1

                    ## now get the track id that corresponds to this track
                    ## to identify the hits in other layers that also belong to the same track
                    seed_ids = evts_ids[i_evt][seed_layers,seed_straws]
                    if (seed_ids==seed_ids[0]).sum() == len(seed_ids):
                        # all hits in seed belong to same track
                        seed_id = seed_ids[0]
                    else: 
                        # oops, at least some hits are false.
                        # could get the track most encountered with eg: mode(seed_ids)
                        # but best to skip this seed from training for now, after we turn off all hits again
                        holder_input[seed_location][n_seeds[seed_location]] = np.zeros(
                            (self.detector.n_layers, 
                             2*self.nstraws_perlayer) )
                        continue
                    track_layers,track_straws=np.where((evts_ids[i_evt]==seed_id))

                    ## now turn on the input for all hits from different modules in the event
                    ## and also at the output for those hits from the same track
                    not_module_mask = (allhits_layers<4*i_module) | (allhits_layers>=4*i_module+4)
                    ilayers,istraws=allhits_layers[not_module_mask], allhits_straws[not_module_mask] # this excludes hits from the same module as the seed 
                    for i,ilayer in enumerate(ilayers):
                        istraw=istraws[i]
                        ihit_index = np.random.choice( np.array( np.where(ihit_perlayer[ilayer]==0)[0] ))
                        holder_input[seed_location][n_seeds[seed_location]][ilayer][ihit_index] = istraw+1
                        if evts_ids[i_evt][ilayer][istraw]==seed_id:
                            holder_target[seed_location][n_seeds[seed_location]][ilayer][ihit_index+1] = 1
                        ## increment i_hit in this layer
                        ihit_perlayer[ilayer][ihit_index]=1
                    
                    ## finally for layers that didn't have a track hit, set the target to the 0th feature
                    for ilayer in range(self.detector.n_layers):
                        if ilayer not in track_layers:
                            holder_target[seed_location][n_seeds[seed_location]][ilayer][0] = 1

                    ## finished processing this seed, increment
                    n_seeds[seed_location] += 1
                    if (n_seeds[seed_location] % print_freq == 0):
                        print (' Processing seed #%d in detector-%s' % (n_seeds[seed_location], seed_location) )

        print ('\nTotal seeds processed per location: ', n_seeds)
              
        ## Now remove all the extra array dimensions, keep only n_seeds
        for seed_location in ['front','middle','back']:
            self.train_input[seed_location] = holder_input[seed_location][:n_seeds[seed_location],:,:]
            self.train_target[seed_location] = holder_target[seed_location][:n_seeds[seed_location],:,:]
        


        
    def prepare_training_data_multiseed(self, evts_hits, evts_ids):
        """
        Prepare training data. 
        Reads in 3D arrays of hits and ids in many event windows.
        The 1st dimension is the event number.
        The hits and ids within each window  are of the same shape: (length,width).
        Within each window generate seeds.
        *** In this iteration, generate multiple inputs per seed, with multiple seeds passed at input ***
        The input will be of shape ([n_seeds*several], length, width)
        """
        #if self.prepared: return
        #self.prepared = True
        
        n_events = evts_hits.shape[0]
        max_seeds = n_events * self.detector.max_tracks * self.detector.n_modules * 5 # guesstimate 5 multi-seed combinations per seed
        print_freq = int(max_seeds / 20)

        holder_input, holder_target={},{}
        n_inputs={} # this will be a counter for model inputs, potentially multiple per seed
        for seed_location in ['front','middle','back']:
            holder_input[seed_location] = np.zeros((int(max_seeds), 
                                                    self.detector.n_layers, 
                                                    2*self.nstraws_perlayer))
            holder_target[seed_location] = np.zeros((int(max_seeds), 
                                                     self.detector.n_layers, 
                                                     self.nstraws_perlayer+1))
            n_inputs[seed_location]=0
    
        ## Iterate over all windows (events)
        for i_evt in range(n_events):

            ## All hits coordinates in this event
            allhits_layers,allhits_straws=np.where(evts_hits[i_evt]>0)

            ## Get all seeds in this event
            seeds_xy, seeds_array=seeding.make_real_seeds(evts_ids[i_evt])
            if len(seeds_xy)==0: continue  
            
            ## Step through each module
            for i_module in range(self.detector.n_modules):

                if i_module<=2: seed_location='front'
                elif i_module<=4: seed_location='middle'
                else: seed_location='back' 
                
                ## Get seeds in this module
                module_seeds_xy = [seed_xy for seed_xy in seeds_xy if (
                    (seed_xy[0][0]>=i_module*4) & (seed_xy[0][0]<i_module*4+4) )]
                if len(module_seeds_xy)==0: continue

                for seed_xy in module_seeds_xy:
                    ## format input for this seed     
                    
                    # array for hits per layer:
                    # this will randomize where hit coordinates appear in the NN input, to avoid bias
                    ihit_perlayer=np.zeros(shape=(self.detector.n_layers,self.nstraws_perlayer)).astype(int)
                                    
                    ## first turn on the seed hits after finding their (layer,straw) location
                    seed_layers,seed_straws = seed_xy
                    for i,ilayer in enumerate(seed_layers):
                        istraw=seed_straws[i]
                        ## get the randomized input index for this hit
                        ihit_index = np.random.choice( np.array( np.where(ihit_perlayer[ilayer]==0)[0] ))
                        ## turn on both hits and seeds elements
                        holder_input[seed_location][n_inputs[seed_location]][ilayer][ihit_index] = istraw+1
                        holder_input[seed_location][n_inputs[seed_location]][ilayer][self.nstraws_perlayer+ihit_index] = 1
                        ## turn on that straw location at the output
                        holder_target[seed_location][n_inputs[seed_location]][ilayer][ihit_index+1] = 1
                        ## increment i_hit in this layer
                        ihit_perlayer[ilayer][ihit_index]=1

                    ## now get the track id that corresponds to this track
                    ## to identify the hits in other layers that also belong to the same track
                    seed_ids = evts_ids[i_evt][seed_layers,seed_straws]
                    if (seed_ids==seed_ids[0]).sum() == len(seed_ids):
                        # all hits in seed belong to same track
                        seed_id = seed_ids[0]
                    else: 
                        # oops, at least some hits are false.
                        # could get the track most encountered with eg: mode(seed_ids)
                        # but best to skip this seed from training for now, after we turn off all hits again
                        holder_input[seed_location][n_inputs[seed_location]] = np.zeros(
                            (self.detector.n_layers, 
                             2*self.nstraws_perlayer) )
                        continue
                    track_layers,track_straws=np.where((evts_ids[i_evt]==seed_id))

                    ## now turn on the input for all hits from different modules in the event
                    ## and also at the output for those hits from the same track
                    not_module_mask = (allhits_layers<4*i_module) | (allhits_layers>=4*i_module+4)
                    ilayers,istraws=allhits_layers[not_module_mask], allhits_straws[not_module_mask] # this excludes hits from the same module as the seed 
                    for i,ilayer in enumerate(ilayers):
                        istraw=istraws[i]
                        ihit_index = np.random.choice( np.array( np.where(ihit_perlayer[ilayer]==0)[0] ))
                        holder_input[seed_location][n_inputs[seed_location]][ilayer][ihit_index] = istraw+1
                        if evts_ids[i_evt][ilayer][istraw]==seed_id:
                            holder_target[seed_location][n_inputs[seed_location]][ilayer][ihit_index+1] = 1
                        ## increment i_hit in this layer
                        ihit_perlayer[ilayer][ihit_index]=1
                    
                    ## finally for layers that didn't have a track hit, set the target to the 0th feature
                    for ilayer in range(self.detector.n_layers):
                        if ilayer not in track_layers:
                            holder_target[seed_location][n_inputs[seed_location]][ilayer][0] = 1
                            
                            
                    ### now form extra inputs from seed combinations
                    ## first get all seeds from the same track
                    extra_seeds_xy=[]
                    for extra_seed_xy in seeds_xy:
                        if extra_seed_xy==seed_xy: continue
                        if evts_ids[i_evt][extra_seed_xy[0][0]][extra_seed_xy[1][0]]==seed_id: extra_seeds_xy+=[extra_seed_xy]
                    n_extra_seeds = len(extra_seeds_xy)
                    
                    if n_extra_seeds==0: 
                        #print('no extra seeds, continuing')
                        n_inputs[seed_location] += 1
                        if (n_inputs[seed_location] % print_freq == 0):
                            print (' Processing seed #%d in detector-%s' % (n_inputs[seed_location], seed_location) )
                        continue
                    if n_extra_seeds==1: extra_seed_id1,extra_seed_id2=0,0
                    else: extra_seed_id1,extra_seed_id2=np.random.choice(n_extra_seeds,size=2, replace=False)
                    #print ('processing %d extra seeds'% n_extra_seeds)
                    #print('2 extra seed ids selected : ', extra_seed_id1, extra_seed_id2)
                    
                    ### set the first extra input with another seed
                    #print('processing 1st seed id ', extra_seed_id1)
                    n_extra_inputs=1
                    holder_input[seed_location][n_inputs[seed_location]+n_extra_inputs] = holder_input[seed_location][n_inputs[seed_location]]
                    ## turn on the extra seed locations
                    extra_seed_layers,extra_seed_straws = extra_seeds_xy[extra_seed_id1]
                    for i,ilayer in enumerate(extra_seed_layers):
                        istraw=extra_seed_straws[i]
                        #print('input for this layer : ', holder_input[seed_location][n_inputs[seed_location]][ilayer])
                        ihit_index = np.where( holder_input[seed_location][n_inputs[seed_location]][ilayer] == istraw+1 )[0]
                        #print('ihit index found : ', ihit_index)
                        holder_input[seed_location][n_inputs[seed_location]+n_extra_inputs][ilayer][self.nstraws_perlayer+ihit_index] = 1
                    ## finally copy the target
                    holder_target[seed_location][n_inputs[seed_location]+n_extra_inputs] = holder_target[seed_location][n_inputs[seed_location]]
                     
                    if n_extra_seeds==1: 
                        #print('continuing after 1 extra seed')
                        n_inputs[seed_location] += 2
                        if (n_inputs[seed_location] % print_freq == 0):
                            print (' Processing seed #%d in detector-%s' % (n_inputs[seed_location], seed_location) )
                        continue
                    ### now set a 3rd input with yet another seed
                    #print('processing 2nd seed id ', extra_seed_id2)
                    n_extra_inputs=2
                    holder_input[seed_location][n_inputs[seed_location]+n_extra_inputs] = holder_input[seed_location][n_inputs[seed_location]+1]
                    ## turn on the extra seed locations
                    extra_seed_layers,extra_seed_straws = extra_seeds_xy[extra_seed_id2]
                    for i,ilayer in enumerate(extra_seed_layers):
                        istraw=extra_seed_straws[i]
                        ihit_index = np.where( holder_input[seed_location][n_inputs[seed_location]][ilayer] == istraw+1 )[0]
                        holder_input[seed_location][n_inputs[seed_location]+n_extra_inputs][ilayer][self.nstraws_perlayer+ihit_index] = 1
                    ## finally copy the target
                    holder_target[seed_location][n_inputs[seed_location]+n_extra_inputs] = holder_target[seed_location][n_inputs[seed_location]]
                    
                    if n_extra_seeds==2: 
                        #print('continuing after 2 extra seeds')
                        n_inputs[seed_location] += 3
                        if (n_inputs[seed_location] % print_freq == 0):
                            print (' Processing seed #%d in detector-%s' % (n_inputs[seed_location], seed_location) )
                        continue
                    ### one last extra input with all seeds from this track
                    #print('processing last extra input with all seeds')
                    n_extra_inputs=3
                    holder_input[seed_location][n_inputs[seed_location]+n_extra_inputs] = holder_input[seed_location][n_inputs[seed_location]+2]
                    ## turn on the extra seed locations
                    for i_extra_seed,extra_seed_xy in enumerate(extra_seeds_xy):
                        if i_extra_seed in [extra_seed_id1, extra_seed_id2]: 
                            #print('skipping extra seed id ', i_extra_seed)
                            continue
                        extra_seed_layers,extra_seed_straws = extra_seed_xy
                        for i_extra_seed_hit,ilayer in enumerate(extra_seed_layers):
                            istraw=extra_seed_straws[i_extra_seed_hit]
                            ihit_index = np.where( holder_input[seed_location][n_inputs[seed_location]][ilayer] == istraw+1 )[0]
                            holder_input[seed_location][n_inputs[seed_location]+n_extra_inputs][ilayer][self.nstraws_perlayer+ihit_index] = 1
                    ## finally copy the target
                    holder_target[seed_location][n_inputs[seed_location]+n_extra_inputs] = holder_target[seed_location][n_inputs[seed_location]]


                    ## finished processing this seed, increment
                    #print('continuing after 3 extra seeds')
                    n_inputs[seed_location] += 4
                    if (n_inputs[seed_location] % print_freq == 0):
                        print (' Processing seed #%d in detector-%s' % (n_inputs[seed_location], seed_location) )


        print ('\nTotal seeds processed per location: ', n_inputs)
              
        ## Now remove all the extra array dimensions, keep only n_seeds
        for seed_location in ['front','middle','back']:
            self.train_input[seed_location] = holder_input[seed_location][:n_inputs[seed_location],:,:]
            self.train_target[seed_location] = holder_target[seed_location][:n_inputs[seed_location],:,:]
        
            

    def fit(self, evts_hits, evts_ids):
        #self.prepare_training_data_NoDF(evts_hits, evts_ids)
        self.prepare_training_data_multiseed(evts_hits, evts_ids)
        print('Starting training...')
        ## callback to optimize when training is stopped
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        min_epochs=self.n_epochs
        for seed_location in ['front','middle','back']:
            self.history[seed_location] = self.model[seed_location].fit(
                self.train_input[seed_location], self.train_target[seed_location],
                batch_size=self.batch_size, epochs=self.n_epochs, callbacks=[callback],
                validation_split=self.val_frac)
            if callback.stopped_epoch<min_epochs: min_epochs=callback.stopped_epoch
        if self.n_epochs>min_epochs: self.n_epochs=min_epochs
      
    def prepare_event_for_prediction(self, evt_hits):
        """
        After the model has been trained, we will want to predict individual events.
        This function processes the 2D arrays of individual events in the proper format for model.predict().
        Reads in the 2D array of hits in the event, of shape: (length,width).
        Generate real seeds from the event hits.
        *** In this iteration, seeds are genrated anywhere in the tracker, and multiple per track ***
        The input for prediction will be of shape (n_seeds, length, width)
        """
        
        ## All hits coordinates in this event
        allhits_layers,allhits_straws=np.where(evt_hits>0)

        ## First form seeds in the event
        seeds_xy, seeds_array=seeding.make_real_seeds(evt_hits)

        ## count seeds
        n_seeds={} 
        for seed_location in ['front','middle','back']:
            n_seeds[seed_location]=0
        module_seeds_xy={}
        ## Step through each module to count seeds
        for i_module in range(self.detector.n_modules):
            if i_module<=2: seed_location='front'
            elif i_module<=4: seed_location='middle'
            else: seed_location='back'
                
            module_seeds_xy[i_module] = [seed_xy for seed_xy in seeds_xy if (
                (seed_xy[0][0]>=i_module*4) & (seed_xy[0][0]<i_module*4+4) )]
            n_seeds[seed_location] += len(module_seeds_xy[i_module])
        
        model_input,i_seed={},{}
        ## Input format
        for seed_location in ['front','middle','back']:
            model_input[seed_location] = np.zeros((n_seeds[seed_location], 
                                                   self.detector.n_layers,  
                                                   2*self.nstraws_perlayer))
            i_seed[seed_location]=0 ## counter for seeds; must equal n_seeds eventually
        
        ## Step through each module to format input for each seed
        for i_module in range(self.detector.n_modules):
            if i_module<=2: seed_location='front'
            elif i_module<=4: seed_location='middle'
            else: seed_location='back'

            ## we already have seeds for each module, iterate through them
            for seed_xy in module_seeds_xy[i_module]:
                ## format input for this seed     

                # array for hits per layer:
                # this will randomize where hit coordinates appear in the NN input, to avoid bias
                ihit_perlayer=np.zeros(shape=(self.detector.n_layers,self.nstraws_perlayer)).astype(int)
                    
                ## first turn on the seed hits after finding their (layer,straw) location
                seed_layers,seed_straws = seed_xy
                for i,ilayer in enumerate(seed_layers):
                    istraw=seed_straws[i]
                    ihit_index = np.random.choice( np.array( np.where(ihit_perlayer[ilayer]==0)[0] ))
                    ## turn on both hits and seeds elements
                    model_input[seed_location][i_seed[seed_location]][ilayer][ihit_index] = istraw+1
                    model_input[seed_location][i_seed[seed_location]][ilayer][self.nstraws_perlayer+ihit_index] = 1
                    ## increment i_hit in this layer
                    ihit_perlayer[ilayer][ihit_index]=1

                ## now turn on all the hits from different modules in the event
                not_module_mask = (allhits_layers<4*i_module) | (allhits_layers>=4*i_module+4)
                ilayers,istraws=allhits_layers[not_module_mask], allhits_straws[not_module_mask] # this excludes hits from the same module as the seed 
                for i,ilayer in enumerate(ilayers):
                    istraw=istraws[i]
                    ihit_index = np.random.choice( np.array( np.where(ihit_perlayer[ilayer]==0)[0] ))
                    model_input[seed_location][i_seed[seed_location]][ilayer][ihit_index] = istraw+1
                    ## increment i_hit in this layer
                    ihit_perlayer[ilayer][ihit_index]=1
                    
                ## finished processing this seed, increment
                i_seed[seed_location] += 1
            
        return model_input
  


    def predict_seed(self, seed_input, seed_location):
        """
        Predict seed for the given model_input, ie single input/prediction.
        Format output in same format as evt_hits, ie shape=(n_layers,n_straws)
        """
        
        ## model output, of shape (n_layers, nstraws_perlayer)
        model_output = tracker_NN.model[seed_location].predict(seed_input)
        
        seed_prediction = np.zeros((self.detector.n_layers, self.detector.n_straws))
        
        for ilayer in range(self.detector.n_layers):
            for ihitstraw in range(self.nstraws_perlayer):
                istraw = seed_input[ilayer][ihitstraw]
                seed_prediction[ilayer][istraw] = model_output[ilayer][ihitstraw+1]
        
        return seed_prediction
    

    def predict_event(self, evt_hits): 
        """
        Read in event (time window) with all hits.
        Return full prediction for all seeds in event.
        """
        model_input = self.prepare_event_for_prediction(evt_hits)
        #print('model_input: \n', model_input)

        n_seeds,model_output,event_prediction={},{},{}
        for seed_location in ['front','middle','back']:
            n_seeds[seed_location]=model_input[seed_location].shape[0]
            print('%d seeds in %s' % (n_seeds[seed_location],seed_location))
            if n_seeds[seed_location]==0:
                event_prediction[seed_location]=np.array([])
                continue
            ## get model output for all seeds
            model_output[seed_location] = self.model[seed_location].predict(model_input[seed_location])
            #print('model_output: \n', model_output[seed_location])
            print(model_output[seed_location].shape)

            ## package output into full prediction, in 32x32 array
            event_prediction[seed_location]=np.zeros( (n_seeds[seed_location],
                                                       self.detector.n_layers, self.detector.n_straws))
            for i_seed in range(n_seeds[seed_location]):
                for ilayer in range(self.detector.n_layers):
                    for ihitstraw in range(self.nstraws_perlayer):
                        istraw = model_input[seed_location][i_seed][ilayer][ihitstraw].astype(int)-1
                        event_prediction[seed_location][i_seed][ilayer][istraw] = model_output[seed_location][i_seed][ilayer][ihitstraw+1]

        return event_prediction

