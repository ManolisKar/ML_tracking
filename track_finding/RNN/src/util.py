import numpy as np
import matplotlib.pyplot as plt


def display_event(detector, straw_hits=None, track_ids=None, figsize=(15, 8)):
    """
    Draw hits and tracks from a single event (time window) in detector.
    Parameters:
        detector: The DetectorGeometry object
        straw_hits: 2D array with the straws that are hit in the event
        track_ids: The track identifier corresponding to each hit
    """

    fig, ax = plt.subplots(figsize=figsize) # note we must use plt.subplots, not plt.subplot
    
    full_width = detector.width
    ax.set_xlim((-0.1*detector.length, 1.1*detector.length))
    ax.set_ylim((-0.1*full_width, 1.1*full_width))

    ## plot a circle at the location of each straw
    for i_layer in range(detector.n_layers):
        for i_straw in range(detector.n_straws):
            if (detector.x[i_layer]==0) or (detector.y[i_layer][i_straw]==0): continue
            circle = plt.Circle((detector.x[i_layer], detector.y[i_layer][i_straw]), detector.straw_radius, color='black', fill=False)
            ax.add_patch(circle)
    
    if straw_hits is not None:
        ## plot all hits
        layers,straws=np.where(straw_hits==1)
        ax.scatter(detector.x[layers],detector.y[layers,straws], marker='.', color='black')

    if track_ids is not None:
        ## plot all tracks with a separate color
        for itrack in range(1,int(np.max(track_ids))+1):
            layers,straws=np.where(track_ids==itrack)
            plt.scatter(detector.x[layers],detector.y[layers,straws], marker='o', label='track %d'%itrack)
        
        ## plot unassigned hits if they exist
        if -1 in track_ids:
            layers,straws=np.where(track_ids==-1)
            plt.scatter(detector.x[layers],detector.y[layers,straws], marker='o', color='black', label='unassigned hits')

    plt.legend(loc=1)
    plt.show()


    

def draw_train_history(history, draw_val=True, figsize=(12,5)):
    """Make plots of training and validation losses and accuracies"""
    for seed_location in ['front','middle','back']:
        plt.figure(figsize=figsize)
        # Plot loss
        plt.subplot(121)
        plt.plot(history[seed_location].epoch, history[seed_location].history['loss'], label='Training set')
        if draw_val:
            plt.plot(history[seed_location].epoch, history[seed_location].history['val_loss'], label='Validation set')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.legend()
        plt.subplot(122)
        plt.plot(history[seed_location].epoch, history[seed_location].history['accuracy'], label='Training set')
        if draw_val:
            plt.plot(history[seed_location].epoch, history[seed_location].history['val_accuracy'], label='Validation set')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim((0, 1))
        plt.title('Training accuracy')
        plt.legend(loc=0)
        plt.tight_layout()




def plot_event_NNoutput(evt_hits, evt_ids=False, selection_threshold=0.6):
    ## make passing ids optional
    
    ## Display event
    display_event(tracker,evt_hits)


    ## We could get prediction simply with predict_event, 
    ## but we also need the model input to compare against truth and seeds
    #model_prediction = tracker_NN.predict_event(evt_hits)
    ## prepare event for prediction 
    model_input = tracker_NN.prepare_event_for_prediction(evt_hits)
    ## iterate over all seed locations
    n_seeds,model_output,model_prediction={},{},{}
    for seed_location in ['front','middle','back']:
        n_seeds[seed_location]=model_input[seed_location].shape[0]
        print('%d seeds in %s' % (n_seeds[seed_location],seed_location))
        if n_seeds[seed_location]==0:
            model_prediction[seed_location]=np.array([])
            continue
        ## get model output for all seeds
        model_output[seed_location] = tracker_NN.model[seed_location].predict(model_input[seed_location])
        ## package output into full prediction, in 32x32 array
        model_prediction[seed_location]=np.zeros( (n_seeds[seed_location],
                                                   tracker.n_layers, tracker.n_straws))
        for i_seed in range(n_seeds[seed_location]):
            for ilayer in range(tracker.n_layers):
                for ihitstraw in range(tracker_NN.nstraws_perlayer):
                    istraw = model_input[seed_location][i_seed][ilayer][ihitstraw].astype(int)
                    model_prediction[seed_location][i_seed][ilayer][istraw] = model_output[seed_location][i_seed][ilayer][ihitstraw+1]
    
    model_prediction_selected={}
    for seed_location in ['front','middle','back']:
        if n_seeds[seed_location]==0: continue
        model_prediction_selected[seed_location] = np.zeros_like(model_prediction[seed_location])
        model_prediction_selected[seed_location][np.where(model_prediction[seed_location]>selection_threshold)] = 1
    n_seeds_total = n_seeds['front']+n_seeds['middle']+n_seeds['back']

    
    ## Now plot input and prediction per seed
    full_detector_width = tracker_NN.detector.width
    ## put all straw locations in a df
    layer_v, straw_v, x_v, y_v =[],[],[],[]
    for ilayer in range(tracker_NN.detector.n_layers):
        for istraw in range(tracker_NN.detector.n_straws):
            layer_v+= [ilayer]
            straw_v+= [istraw]
            x_v+= [tracker_NN.detector.x[ilayer] ]
            y_v+= [tracker_NN.detector.y[ilayer][istraw] ]
    df = pd.DataFrame({'layer':layer_v, 'straw':straw_v, 'x':x_v, 'y':y_v})

    
    ## Step through all seeds from each location and plot model prediction
    for seed_location in ['front','middle','back']:
        for i_seed in range(n_seeds[seed_location]):
            print('seed %d in detector %s' % (i_seed, seed_location))
            ## Plot the raw NN output, which is the probability for each straw
            ## that it belongs to the same track as the seed
            fig, axs = plt.subplots(figsize=(20,8))
            axs.set_title('NN raw output for seed #%d in %s' % (i_seed,seed_location))

            axs.set_xlim((-0.1*tracker_NN.detector.length, 1.1*tracker_NN.detector.length))
            axs.set_ylim((-0.1*full_detector_width, 1.1*full_detector_width))
            # plot a circle at the location of each straw
            for ilayer,straw_x in enumerate(tracker_NN.detector.x):
                for straw_y in tracker_NN.detector.y[ilayer]:
                    circle = plt.Circle((straw_x, straw_y), tracker_NN.detector.straw_radius, 
                                        color='black', fill=False)
                    axs.add_patch(circle)
            ix,iy=np.where(model_prediction[seed_location][i_seed]>0)
            axs.scatter(tracker_NN.detector.x[ix],tracker_NN.detector.y[ix,iy],
                        c=model_prediction[seed_location][i_seed][ix,iy], cmap = 'RdPu', alpha =0.5)
            plt.show()

            
            ### Plot the NN output values corresponding to the straws that are predicted to belong to the track
            fig, axs = plt.subplots()      
            axs.set_title('NN output for straws that are predicted to belong to the track')
            hits_x,hits_y=np.where(model_prediction_selected[seed_location][i_seed]==1)
            axs.hist(model_prediction[seed_location][i_seed][hits_x,hits_y])
            plt.show()
            print ('RMS = %.4f' % np.std(model_prediction[seed_location][i_seed][hits_x,hits_y]))
            
                               
            ## If we haven't passed the truth, then we're done (for this seed)
            if evt_ids is False: continue

            ### Now plot relative to the truth, if it is passed
            ## Get the layers/straws for this seed
            seed_layers,seed_hits = np.where(model_input[seed_location][i_seed][:,tracker_NN.nstraws_perlayer:] > 0)
            seed_straws = model_input[seed_location][i_seed][seed_layers.astype(int),seed_hits.astype(int)]

            ## now get the track id that corresponds to this track
            seed_ids = evt_ids[seed_layers,seed_straws.astype(int)]
            if (seed_ids==seed_ids[0]).sum() == len(seed_ids):
            # all hits in seed belong to same track
                seed_id = seed_ids[0]
            else: 
                # oops, at least some hits are false.
                # could get the track most encountered with eg: mode(seed_ids)
                # but best to skip this seed for now
                print('\n Unpure seed! We\'ll just use the seed id of the first hit as truth.')
                seed_id = seed_ids[0]
                                

            ### Plot the NN output values corresponding to the straws that belong to the track
            fig, axs = plt.subplots()    
            axs.set_title('NN output for straws that belong to the track')
            hits_x,hits_y=np.where(evt_ids==seed_id)
            axs.hist(model_prediction[seed_location][i_seed][hits_x,hits_y])
            plt.show()
            #print ('RMS = %.4f' % np.std(model_prediction[seed_location][i_seed][hits_x,hits_y]))


            ### Plot the NN output values corresponding to the straws that are predicted to belong to the track,
            ### but actually don't
            fig, axs = plt.subplots()        
            axs.set_title("NN output for straws that are predicted to belong to the track, but don't")
            badpred_x,badpred_y=np.where( (model_prediction_selected[seed_location][i_seed]==1) & (evt_ids!=seed_id))
            axs.hist(model_prediction[seed_location][i_seed][badpred_x,badpred_y])
            plt.show()
            #print ('RMS = %.4f' % np.std(model_prediction[seed_location][i_seed][badpred_x,badpred_y]))            



            
def raw_score(tracker_NN, evts_hits, evts_ids, selection_threshold=0.7, draw_output=False):
    ## Accuracy score for the raw NN output, before any disambiguation step.
    ## Scored per input, how many of the predicted hits actually do belong in the searched track. 
    ## Predicted here just means, those hits with output above the selection threshold.    
    
    ## counters
    n_seeds_unpure,n_seeds_total =0,0
    n_track_hits, n_track_correct, n_predicted, n_correct, n_wrong = {},{},{},{},{}
    track_correct, predicted_correct, predicted_wrong = {},{},{}
    for seed_location in ['front','middle','back']:
        n_track_hits[seed_location]=0
        n_track_correct[seed_location]=0
        n_predicted[seed_location]=0
        n_correct[seed_location]=0
        n_wrong[seed_location]=0
        track_correct[seed_location]=[]
        predicted_correct[seed_location]=[]
        predicted_wrong[seed_location]=[]

    ## iterate over all events
    n_evts=evts_hits.shape[0]
    n_print=int(n_evts/10)
    for i_evt in range(n_evts):
        #if (i_evt%n_print)==0: print('scoring event #%d'%i_evt)
    
        ## prepare event for prediction 
        model_input = tracker_NN.prepare_event_for_prediction(evts_hits[i_evt])
        
        ## iterate over all seed locations
        model_output,event_prediction={},{}
        for seed_location in ['front','middle','back']:
            n_seeds_loc=model_input[seed_location].shape[0]
            if n_seeds_loc==0:
                continue
            n_seeds_total+=n_seeds_loc
            ## get model output for all seeds here
            model_output[seed_location] = tracker_NN.model[seed_location].predict(model_input[seed_location])
            
            ## iterate over all seeds in this location
            for i_seed in range(n_seeds_loc):
            
                ## Get the layers/straws for this seed
                seed_layers,seed_hits = np.where(model_input[seed_location][i_seed][:,tracker_NN.nstraws_perlayer:] > 0)
                seed_straws = model_input[seed_location][i_seed][seed_layers.astype(int),seed_hits.astype(int)]-1
                
                ## now get the track id that corresponds to this track
                seed_ids = evts_ids[i_evt][seed_layers,seed_straws.astype(int)]
                if (seed_ids==seed_ids[0]).sum() == len(seed_ids):
                # all hits in seed belong to same track
                    seed_id = seed_ids[0]
                else: 
                    # oops, at least some hits are false.
                    # could get the track most encountered with eg: mode(seed_ids)
                    # but best to skip this seed for now
                    n_seeds_unpure+=1
                    continue
                track_layers,track_straws=np.where((evts_ids[i_evt]==seed_id))
                nhits_track = len(track_layers)
                n_track_hits[seed_location]+=nhits_track
                
                ## get all hits selected by model with this seed
                selected_layers,selected_ids= np.where(model_output[seed_location][i_seed][:,1:] > selection_threshold )
                selected_straws = model_input[seed_location][i_seed][selected_layers,selected_ids]-1
                nhits_predict = len(selected_layers)
                n_predicted[seed_location]+=nhits_predict
                
                ## Now get the track layers/straws that were correctly predicted
                nhits_correct=0
                for i,ilayer in enumerate(track_layers):
                    if ((ilayer in selected_layers) and 
                        (selected_straws[np.where(selected_layers==ilayer)][0] == track_straws[i])):
                        nhits_correct+=1
                accuracy_score = nhits_correct/nhits_track
                track_correct[seed_location]+=[accuracy_score]
                n_track_correct[seed_location]+=nhits_correct
                
                ## Now count correct/wrong out of the predicted hits
                nhits_correct,nhits_wrong=0,0
                for i,ilayer in enumerate(selected_layers):
                    if ((ilayer in track_layers) and 
                        (track_straws[np.where(track_layers==ilayer)][0] == selected_straws[i])):
                        nhits_correct+=1
                    else:
                        nhits_wrong+=1
                n_correct[seed_location]+=nhits_correct
                n_wrong[seed_location]+=nhits_wrong
                predicted_correct[seed_location]+=[nhits_correct/nhits_predict]
                predicted_wrong[seed_location]+=[nhits_wrong/nhits_predict]
          
    ## Processed all seeds. Final results:
    print("Total unpure seeds found :: %d (out of %d)" % (n_seeds_unpure,n_seeds_total))
    raw_scores={}
    for seed_location in ['front','middle','back']:
        print('For %s seeds:'%seed_location)
        accuracy_all=100*n_track_correct[seed_location]/n_track_hits[seed_location]
        error_returned=100*n_wrong[seed_location]/n_predicted[seed_location]
        print('Accurately found fraction of track hits :: ', accuracy_all)
        print('Accurately predicted fraction of found hits :: ', 100*n_correct[seed_location]/n_predicted[seed_location])
        print('Wrongly predicted fraction of found tracks :: ', error_returned)
        raw_scores[seed_location]=[accuracy_all,error_returned]
        if draw_output:
            plt.hist(track_correct[seed_location],bins=30)
            plt.show()
            plt.hist(predicted_correct[seed_location],bins=30)
            plt.show()
            plt.hist(predicted_wrong[seed_location],bins=30)
            plt.show()
    
    return raw_scores
                

            
def score_function(true_ids, predicted_ids, verbose=False):
    '''
    Calculate an accuracy score based on hits within events, which belong to seeded tracks.
    Both true and predicted ids are of shape (n_evts, length, width).
    Accuracy score is calculated as the fraction of correctly assigned hits.
    '''
    print('Full prediction scoring')
    
    n_trackhits, n_trackhits_correct=0,0
    n_predicted,n_predicted_correct=0,0
    
    n_evts=true_ids.shape[0]
    n_print=max(1,int(n_evts/10))
    for i_evt in range(n_evts):
        #if (i_evt%n_print)==0: print('scoring event #%d'%i_evt)
        
        ## get all true track ids in this event
        track_ids = [i for i in np.unique(true_ids[i_evt]) if i>0]
        if len(track_ids)==0: continue
        ## get all predicted ids in this event
        pred_ids = [i for i in np.unique(predicted_ids[i_evt]) if i>0]
        
        for pred_id in pred_ids:
            # get all hits that are predicted to be in the same track
            pred_layers,pred_straws=np.where(predicted_ids[i_evt]==pred_id)
            n_predicted += len(pred_layers)
            
            # get the real track that corresponds to this (ambiguity?)
            trackhit_ids, counts = np.unique(true_ids[i_evt][pred_layers,pred_straws], return_counts=True)
            most_id=np.argmax(counts)
            track_id = trackhit_ids[most_id]
            
            ''' ** Alt way to get track_id: **
            track_id=0
            for idx in track_ids:
                layers,straws=np.where(true_ids[i_evt]==idx)
                n_hits=len(layers)
                if np.sum(predicted_ids[i_evt][layers,straws])>(n_hits/2.):
                    n_trackhits+=n_hits
                    track_id=idx
                    break
            '''
            
            # get how many of the predicted_ids are correct:
            n_predicted_correct += len( np.where( true_ids[i_evt][pred_layers,pred_straws]==track_id )[0] )
            
            ## Hits in the real track:
            layers,straws=np.where(true_ids[i_evt]==track_id)
            n_trackhits += len(layers)
            
            # how many of these are predicted correctly
            n_trackhits_correct += len( np.where( predicted_ids[i_evt][layers,straws]==pred_id )[0] )
                                
    if n_trackhits==0:
        print('No tracks found in events passed')
        return 0
    if n_predicted==0:
        print('No predicted tracks found in events passed')
        return 0
    
    accuracy_full=100*n_trackhits_correct/n_trackhits
    error_returned=1-n_predicted_correct/n_predicted
    print('Fraction of track hits identified correctly: ', accuracy_full)
    print('Fractional error (wrongly assigned hits): ', error_returned)
    
    return accuracy_full,error_returned


