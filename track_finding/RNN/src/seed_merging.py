### Identify "golden" seeds that give great track candidates.
### Assume that all their hits are correct, therefore:
### * merge their seeds with other seeds that were included in the golden track hits
### * remove golden track hits from other seeds input

import numpy as np

def merge_golden_seeds(tracker_NN, model_input, #looseness, #model_prediction=np.array(0),
                       removed_seed_ids, good_track_ids, verbose=False ):    
    if verbose: print('Searching for golden seeds')
    
    selection_threshold = 0.6# + looseness*0.07    
    #model_input = tracker_NN.prepare_event_for_prediction(evt_hits)

    n_seeds,model_output,model_prediction={},{},{}
    for seed_location in ['front','middle','back']:
        n_seeds[seed_location]=model_input[seed_location].shape[0]
        model_prediction[seed_location]=np.zeros( (n_seeds[seed_location],
                                                   tracker_NN.detector.n_layers, tracker_NN.detector.n_straws))
        if n_seeds[seed_location]==0: continue
        ## get model output for all seeds here
        model_output[seed_location] = tracker_NN.model[seed_location].predict(model_input[seed_location])
        ## package output into full prediction, in 32x32 array
        for i_seed in range(n_seeds[seed_location]):
            for ilayer in range(tracker_NN.detector.n_layers):
                for ihitstraw in range(tracker_NN.nstraws_perlayer):
                    istraw = model_input[seed_location][i_seed][ilayer][ihitstraw].astype(int)-1
                    model_prediction[seed_location][i_seed][ilayer][istraw] = model_output[seed_location][i_seed][ilayer][ihitstraw+1]

    n_seeds_total=n_seeds['front']+n_seeds['middle']+n_seeds['back']
    if n_seeds_total==0:
        if verbose: print('\nNo seeds in model input')
        return model_input, np.nan, 0, 0
    ## assign an ID to each seed, so that we can combine and delete them
    seed_ids=np.arange(n_seeds_total)
    
    
    ### Stack all seed locations into single array, for ease of seed counting and id'ing
    model_prediction_full = np.concatenate( (model_prediction['front'],model_prediction['middle'],model_prediction['back']) )
    model_input_full = np.concatenate( (model_input['front'],model_input['middle'],model_input['back']) )
    model_prediction_selected = np.zeros_like(model_prediction_full)
    model_prediction_selected[np.where( model_prediction_full>selection_threshold )] = 1
    
    
    ## indicator whether input was modified
    modified_input=0
    new_good_ids=[]
    good_prediction_mins=[]
    ## go through all seeds, search for "golden" tracks
    for seed_id in seed_ids:
        if seed_id in removed_seed_ids: continue
        if verbose: print('processing seed ', seed_id)

        ## Get the layers/straws for this seed
        seed_layers,seed_hits = np.where(model_input_full[seed_id][:,tracker_NN.nstraws_perlayer:] > 0)
        seed_straws = model_input_full[seed_id][seed_layers,seed_hits.astype(int)]-1
        ## Get the hits that are predicted to belong with this seed
        predicted_layers,predicted_straws=np.where(model_prediction_selected[seed_id]==1)
        n_predicted_hits = len(predicted_layers)
        if n_predicted_hits<7: 
            if verbose: print(' -- Less than 7 predicted hits, skipping seed ')
            continue

        prediction_min = min(    model_prediction_full[seed_id][predicted_layers,predicted_straws] )
        prediction_rms = np.std( model_prediction_full[seed_id][predicted_layers,predicted_straws] )
        nhits_track    = np.sum( model_prediction_selected[seed_id][predicted_layers,predicted_straws] )

        if verbose:
            print('min prediction = ', prediction_min)
            print('rms prediction = ', prediction_rms)
            print('nhits_track = ', nhits_track)

        golden_threshold = 0.98# - looseness*0.02
        if (prediction_min>golden_threshold):# & (prediction_rms<0.005) & (nhits_track>=6):
            ## Very good track, likely with all predicted hits correct            
            if not (seed_id in good_track_ids):
                if verbose: print('Accepting GOLDEN track with seed id #%d\n'%seed_id)
                new_good_ids += [seed_id]
                good_prediction_mins += [prediction_min]

    new_good_ids=np.array(new_good_ids)
    good_prediction_mins=np.array(good_prediction_mins)
    if verbose: 
        print('newly found good seeds : ', new_good_ids)
        print('their predictions mins : ', good_prediction_mins)
    for seed_id in new_good_ids:
        if seed_id in removed_seed_ids: continue
        if seed_id in good_track_ids: continue            
        if verbose: print('processing new good seed ', seed_id)
        max_prediction_min = good_prediction_mins[np.where(new_good_ids==seed_id)[0][0]]
        best_seed_id=seed_id
        merged_seeds=[seed_id]
        if verbose: print('best seed, max prediction min, [merged seeds] : ', best_seed_id, max_prediction_min, merged_seeds)
        ## Get the layers/straws for this seed
        seed_layers,seed_hits = np.where(model_input_full[seed_id][:,tracker_NN.nstraws_perlayer:] > 0)
        seed_straws = model_input_full[seed_id][seed_layers,seed_hits.astype(int)]-1
        ## Get the hits that are predicted to belong with this seed
        predicted_layers,predicted_straws=np.where(model_prediction_selected[seed_id]==1)
        n_predicted_hits = len(predicted_layers)

        ## Step through all other seeds, search for ones that belong to the same track
        for seed_id2 in seed_ids:
            if seed_id2 == seed_id: continue
            if seed_id2 in removed_seed_ids: continue

            seed_layers2,seed_hits2 = np.where(model_input_full[seed_id2][:,tracker_NN.nstraws_perlayer:] > 0)
            seed_straws2 = model_input_full[seed_id2][seed_layers2,seed_hits2.astype(int)]-1              
            if verbose>1:
                print('model input for seed_id %d : \n'%seed_id2,model_input_full[seed_id2])
                print('seed layers,straws, hits:')
                print(seed_layers2)
                print(seed_straws2)
                print(seed_hits2)
            layers2,hits2=np.where(model_input_full[seed_id2][:,:tracker_NN.nstraws_perlayer]>0)
            straws2=model_input_full[seed_id2][layers2,hits2.astype(int)]-1

            ## number of seed hits of seed_id2 that were selected in seed_id
            n_seed_same = np.sum( model_prediction_selected[seed_id][seed_layers2,seed_straws2.astype(int)] )
            ## number and fraction of common selected hits overall
            n_hits_same = np.sum( model_prediction_selected[seed_id2][predicted_layers,predicted_straws] )
            frac_hits_same = n_hits_same / n_predicted_hits

            if verbose: 
                print('Cross-checking with seed #',seed_id2)
                print('%d same seed hits '%n_seed_same)
                print('%d same hits overall '%n_hits_same)
                print('%dpc fractional same hits '%(100*frac_hits_same))

            if (n_seed_same>=3) & (n_hits_same>=6):
                ## Since the original seed is high purity, assume that this seed is part of the same track.
                ## Remove this seed_id2 from further consideration
                ## -- trying to be a little conservative in the requirements to remove a seed
                if verbose: print('Merging seed#%d into seed#%d'%(seed_id2,seed_id))
                merged_seeds+=[seed_id2]
                if seed_id2 in new_good_ids:
                    if good_prediction_mins[np.where(new_good_ids==seed_id2)[0][0]]>max_prediction_min:
                        max_prediction_min = good_prediction_mins[np.where(new_good_ids==seed_id2)[0][0]]
                        best_seed_id=seed_id2
        
        ## consolidate the merged seeds into the one with the best output
        if verbose: print('\n found best seed id : ', best_seed_id)
        for merged_seed_id in merged_seeds:
            if merged_seed_id!=best_seed_id:
                removed_seed_ids+=[merged_seed_id]
                if verbose>1: print('removing merged seed %d with input:\n '%merged_seed_id, model_input_full[merged_seed_id])
                ## turn on the seed indicator for this seed on the consolidated input
                seed_layers2,seed_hits2 = np.where(model_input_full[merged_seed_id][:,tracker_NN.nstraws_perlayer:] > 0)
                seed_straws2 = model_input_full[merged_seed_id][seed_layers2,seed_hits2.astype(int)]-1              
                ## note that hit_location is different at input of each seed
                for i,seed_layer2 in enumerate(seed_layers2):
                    seed_straw2=seed_straws2[i]
                    if (seed_straw2+1) not in model_input_full[best_seed_id][seed_layer2]: continue
                    seed_hit2=np.where(model_input_full[best_seed_id][seed_layer2]==seed_straw2+1)[0][0]
                    model_input_full[best_seed_id][seed_layer2][tracker_NN.nstraws_perlayer+seed_hit2] = 1
            if verbose>1: print('model input for best seed after this merge: \n', model_input_full[best_seed_id])

        good_track_ids += [best_seed_id]

        ## now remove the hits of the good track from other inputs

        ## We assume that all its predicted hits are correct here, and remove them from all other inputs
        ## - we need to do this only after we look for shared hits with other seeds to merge
        ## - should we tighten the requirement here, eg remove only hits that are selected by 2 merged seeds
        ## Let's try to remove only the very highest scoring hits, even higher than the golden thresh
        golden_layers,golden_straws = np.where(model_prediction_full[seed_id][:,:] > 0.995)
        if verbose: print('golden layers and straws : \n', golden_layers, golden_straws)
        for seed_id2 in seed_ids:
            if (seed_id2 == best_seed_id) or (seed_id2 in removed_seed_ids): continue                
            layers2,hits2=np.where(model_input_full[seed_id2][:,:tracker_NN.nstraws_perlayer]>0)
            straws2=model_input_full[seed_id2][layers2,hits2.astype(int)]-1
            if verbose: print('layers and straws for this seed : \n', layers2, straws2)
            for layer2 in np.unique(layers2):
                if seed_id2 in removed_seed_ids: continue                
                if layer2 not in golden_layers: continue
                i_gold=np.where(golden_layers==layer2)[0][0]
                if golden_straws[i_gold] not in straws2[layers2==layer2]: continue
                if verbose>1:
                    print('model_input for seed %d layer %d before hit removal : '%(seed_id2,layer2), model_input_full[seed_id2][layer2])
                golden_straw = golden_straws[ np.where(golden_layers==layer2)[0][0] ]
                ihit = np.where(model_input_full[seed_id2][layer2]==golden_straw+1)[0][0]
                if verbose>1:
                    print('golden layer/straw/hit : ', layer2, golden_straw, ihit)
                    print('Removing golden hit (%d,%d) at loc%d from seed %d' % (layer2,int(golden_straw),ihit,seed_id2))
                model_input_full[seed_id2][layer2][ihit] = 0
                if verbose>1: 
                    print('model_input for that seed layer after removal : ', model_input_full[seed_id2][layer2])
                if ( (model_input_full[seed_id2][layer2][ihit+tracker_NN.nstraws_perlayer]==1) 
                    and (seed_id2 not in removed_seed_ids) ):
                    removed_seed_ids += [seed_id2]
                    if verbose: print('removing seed ',seed_id2)
                modified_input=1

        if verbose:
            print(' Good track identified:: ')
            display_event(tracker, straw_hits=model_prediction_selected[best_seed_id])        

    ## Recover partial inputs from each seed location
    model_input['front'] = model_input_full[:n_seeds['front']]
    model_input['middle'] = model_input_full[n_seeds['front']:n_seeds['front']+n_seeds['middle']]
    model_input['back'] = model_input_full[n_seeds['front']+n_seeds['middle']:]
    
    return model_input, removed_seed_ids, good_track_ids, modified_input



### Identify "silver" seeds that give good track candidates, but don't assume that all their hits are correct.
### Call with increasingly loose criteria.
### * merge with other seeds that have enough shared hits
### * remove the most reliable silver hits from other seeds input 
###   ** (provided they don't also have high output in other seeds)


def merge_silver_seeds(tracker_NN, model_input, looseness, #model_prediction=np.array(0),
                       removed_seed_ids, good_track_ids, verbose=False):    

    if verbose:
        print('\n\n Merging silver seeds\n')
        print('removed_seed_ids : ', removed_seed_ids)
        print('good_track_ids : ', good_track_ids)
        
    n_seeds,model_output,model_prediction={},{},{}
    for seed_location in ['front','middle','back']:
        n_seeds[seed_location]=model_input[seed_location].shape[0]
        model_prediction[seed_location]=np.zeros( (n_seeds[seed_location],
                                                   tracker_NN.detector.n_layers, tracker_NN.detector.n_straws))
        if n_seeds[seed_location]==0: continue
        ## get model output for all seeds here
        model_output[seed_location] = tracker_NN.model[seed_location].predict(model_input[seed_location])
        ## package output into full prediction, in 32x32 array
        for i_seed in range(n_seeds[seed_location]):
            for ilayer in range(tracker_NN.detector.n_layers):
                for ihitstraw in range(tracker_NN.nstraws_perlayer):
                    istraw = model_input[seed_location][i_seed][ilayer][ihitstraw].astype(int)-1
                    model_prediction[seed_location][i_seed][ilayer][istraw] = model_output[seed_location][i_seed][ilayer][ihitstraw+1]
                    
    n_seeds_total=n_seeds['front']+n_seeds['middle']+n_seeds['back']
    if n_seeds_total==0:
        print('\nNo seeds in model input' )
        return model_input, np.nan, 0, 0
    ## assign an ID to each seed, so that we can combine and delete them
    seed_ids=np.arange(n_seeds_total)    
    
    ## Selection based on prediction
    selection_threshold=0.6# + looseness*0.05
    ### Stack all seed locations into single array, for ease of seed counting and id'ing
    model_prediction_full = np.concatenate( (model_prediction['front'],model_prediction['middle'],model_prediction['back']) )
    model_input_full = np.concatenate( (model_input['front'],model_input['middle'],model_input['back']) )
    model_prediction_selected = np.zeros_like(model_prediction_full)
    model_prediction_selected[np.where( model_prediction_full>selection_threshold )] = 1
    

    ## indicator whether input was modified
    modified_input=0
    ### Go through all seeds, search for "golden" tracks
    for seed_id in seed_ids:
        if seed_id in removed_seed_ids: continue
        if verbose: print('processing seed id #', seed_id)

        ## Get the layers/straws for this seed
        seed_layers,seed_hits = np.where(model_input_full[seed_id][:,tracker_NN.nstraws_perlayer:] == 1)
        seed_straws = model_input_full[seed_id][seed_layers,seed_hits.astype(int)]-1
        ## Get the hits that are predicted to belong with this seed
        predicted_layers,predicted_straws=np.where(model_prediction_selected[seed_id]==1)
        n_predicted_hits = len(predicted_layers)
        ## if too few good hits, remove seed
        if n_predicted_hits<6:
            removed_seed_ids += [seed_id]
            modified_input=1
            continue
        ## Get the hits that are too uncertain, ie below a threshold
        uncertain_layers,uncertain_straws=np.where( (model_prediction_full[seed_id]<0.9) & (model_prediction_full[seed_id]>selection_threshold) )
        if verbose: print('uncertain layers, straws ::\n', uncertain_layers, uncertain_straws)
        n_uncertain = len(uncertain_layers)
        ## if too many uncertain hits, remove seed
        if n_uncertain>2: # and looseness>=1
            if verbose: print('removing seed, %d uncertain hits' % n_uncertain)
            removed_seed_ids += [seed_id]
            modified_input=1
            continue
        if looseness>0:
            ## turn off the uncertain hits in later iterations
            for i,uncertain_layer in enumerate(uncertain_layers):
                uncertain_straw=uncertain_straws[i]
                if verbose: print('Removing uncertain hit at (%d,%d)' % (uncertain_layer,uncertain_straw))
                if (uncertain_straw+1) not in model_input_full[seed_id][uncertain_layer]:
                    ## likely the hit was just removed by another process
                    continue
                ihit = np.where(model_input_full[seed_id][uncertain_layer]==uncertain_straws[i]+1)[0][0]
                if model_input_full[seed_id][ilayer][ihit+tracker_NN.nstraws_perlayer]==1:
                    ## uncertain seed hit? just remove the seed
                    if verbose: print(' removing uncertain seed hit')
                    removed_seed_ids += [seed_id]
                    continue
                model_input_full[seed_id][ilayer][ihit] = 0
                if verbose>1: print('model input after removal : \n', model_input_full[seed_id])

        prediction_min = min(    model_prediction_full[seed_id][predicted_layers,predicted_straws] )
        prediction_rms = np.std( model_prediction_full[seed_id][predicted_layers,predicted_straws] )
        nhits_track    = np.sum( model_prediction_selected[seed_id][predicted_layers,predicted_straws] )
        if verbose:
            print('min prediction = ', prediction_min)
            print('rms prediction = ', prediction_rms)
            print('nhits_track = ', nhits_track)
  
        ### Turn off all the hits that had output below a cutoff (increasing with looseness)
        ### from the input of this same seed  
        hit_layers,hit_ids = np.where(model_input_full[seed_id][:,:tracker_NN.nstraws_perlayer] > 0)
        hit_straws = model_input_full[seed_id][hit_layers,hit_ids.astype(int)]-1
        for i,hit_layer in enumerate(hit_layers):
            hit_straw=hit_straws[i].astype(int)
            hit_id=hit_ids[i]
            ## Check if the corresponding output is less than a cutoff
            cutoff=0.1+looseness*0.1
            if (model_prediction_full[seed_id][hit_layer][hit_straw]<cutoff):
                if verbose: print('Cutoff: Removing hit at (%d,%d)' % (hit_layer,hit_straw))
                model_input_full[seed_id][hit_layer][hit_id] = 0
                modified_input=1
                
        prediction_min = min(    model_prediction_full[seed_id][predicted_layers,predicted_straws] )
        prediction_rms = np.std( model_prediction_full[seed_id][predicted_layers,predicted_straws] )
        nhits_track    = np.sum( model_prediction_selected[seed_id][predicted_layers,predicted_straws] )
        if verbose:
            print('min prediction = ', prediction_min)
            print('rms prediction = ', prediction_rms)
            print('nhits_track = ', nhits_track)

        silver_threshold = 0.95 - looseness*0.01
        if (prediction_min>silver_threshold):# & (prediction_rms<0.005) & (nhits_track>=6):
            ## Good track, but still some ambiguity
            ## don't assume that all its hits are necessarily correct
            if not (seed_id in good_track_ids):
                if verbose: print('Accepting SILVER track with seed id #%d\n'%seed_id)
                good_track_ids += [seed_id]
                modified_input=1
            
            ## Now step through all other seeds, search for ones that belong to the same track
            merged_seeds=[seed_id]
            for seed_id2 in seed_ids:
                if seed_id2==seed_id: continue
                if seed_id2 in removed_seed_ids: continue

                seed_layers2,seed_hits2 = np.where(model_input_full[seed_id2][:,tracker_NN.nstraws_perlayer:] == 1)
                seed_straws2 = model_input_full[seed_id2][seed_layers2,seed_hits2.astype(int)]-1
                layers2,hits2=np.where(model_input_full[seed_id2][:,:tracker_NN.nstraws_perlayer]>0)
                straws2=model_input_full[seed_id2][layers2,hits2.astype(int)]-1
                if verbose>1:
                    print('model input for seed_id %d : \n'%seed_id2,model_input_full[seed_id2])
                    print('seed layers,straws, hits:')
                    print(seed_layers2)
                    print(seed_straws2)
                    print(seed_hits2)
                ## number of seed hits of seed_id2 that were selected in seed_id
                n_seed_same = np.sum( model_prediction_selected[seed_id][seed_layers2,seed_straws2.astype(int)] )
                ## number and fraction of common selected hits overall
                n_hits_same = np.sum( model_prediction_selected[seed_id2][predicted_layers,predicted_straws] )
                frac_hits_same = n_hits_same / n_predicted_hits

                if verbose: 
                    print('Cross-checking with seed #',seed_id2)
                    print('%d same seed hits '%n_seed_same)
                    print('%d same hits overall '%n_hits_same)
                    print('%dpc fractional same hits '%(100*frac_hits_same))

                if (n_seed_same>=3-looseness*0.1) & (n_hits_same>=7-looseness*0.4):
                    ## Since the original seed is of good quality, we assume that this seed is part of the same track.
                    ## Remove this seed_id2 from further consideration.
                    ## We still don't assume that all its hits are correctly assigned
                    ### TO DO: pick the seed with the best output
                    if verbose: print('Merging seed#%d into seed#%d'%(seed_id2,seed_id))
                    merged_seeds += [seed_id2]
                    removed_seed_ids += [seed_id2]
                    if verbose>1:
                        print('merging seed layers, straws, hits:')
                        print(seed_layers2)
                        print(seed_straws2)
                        print(seed_hits2)
                        print('seed input before merge:\n', model_input_full[seed_id])
                    ## note that hit_location is different at input of each seed
                    for i,seed_layer2 in enumerate(seed_layers2):
                        seed_straw2=seed_straws2[i]
                        if (seed_straw2+1) not in model_input_full[seed_id][seed_layer2]: 
                            ## pretty weird if a seed hit of a seed we're merging with has been removed... 
                            ## just skip for now
                            continue
                        seed_hit2=np.where(model_input_full[seed_id][seed_layer2]==seed_straw2+1)[0][0]
                        model_input_full[seed_id][seed_layer2][tracker_NN.nstraws_perlayer+seed_hit2] = 1
                    if verbose>1:
                        print('seed input after merge:\n', model_input_full[seed_id])
                    modified_input=1

            ### Find the highest quality hits among the merged seeds
            for i,ilayer in enumerate(predicted_layers):
                istraw = predicted_straws[i]
                if (istraw+1) not in model_input_full[seed_id][ilayer]: continue
                ihit = np.where(model_input_full[seed_id][ilayer]==istraw+1)[0][0]
                ## this hit is high-quality if its output is high enough among all merged seeds
                ## or at least nmin merged seeds find it to be high
                high_threshold=0.98 - looseness*0.01
                if np.sum(model_prediction_full[merged_seeds, ilayer, istraw]>high_threshold)==len(merged_seeds):
                    if verbose: print('Good hit at (%d,%d)' % (ilayer,istraw))
                    ## Set it to 0 at other inputs, except if other seeds have selected it with high-ish output also
                    for seed_id2 in seed_ids:
                        if ((seed_id2 in good_track_ids) 
                            or (seed_id2 in removed_seed_ids) ): continue
                        if (istraw+1) not in model_input_full[seed_id2][ilayer][:tracker_NN.nstraws_perlayer]: continue
                        ihit2 = np.where(model_input_full[seed_id2][ilayer][:tracker_NN.nstraws_perlayer]==istraw+1)[0][0]
                        model_input_full[seed_id2][ilayer][ihit2] = 0
                        if model_input_full[seed_id2][ilayer][ihit2+tracker_NN.nstraws_perlayer]==1:
                            if verbose: print('removing seed %d, as it lost a seed hit')
                            removed_seed_ids += [seed_id2]
                        modified_input=1

            if verbose: 
                print(' Good track identified:: ')
                display_event(tracker, straw_hits=model_prediction_selected[seed_id])        
            
    ## Recover partial inputs from each seed location
    model_input['front'] = model_input_full[:n_seeds['front']]
    model_input['middle'] = model_input_full[n_seeds['front']:n_seeds['front']+n_seeds['middle']]
    model_input['back'] = model_input_full[n_seeds['front']+n_seeds['middle']:]

    return model_input, removed_seed_ids, good_track_ids, modified_input




def get_prediction_selected(tracker_NN, model_input, selection_threshold):  

    ## Beware: This function washes away the seed location information.
    ## Apply it only as a last step.
    
    #print('\n Now in get_prediction_selected script\n')
    
    n_seeds,model_output,model_prediction={},{},{}
    for seed_location in ['front','middle','back']:
        n_seeds[seed_location]=model_input[seed_location].shape[0]
        model_prediction[seed_location]=np.zeros( (n_seeds[seed_location],
                                                   tracker_NN.detector.n_layers, tracker_NN.detector.n_straws))
        if n_seeds[seed_location]==0: continue
        ## get model output for all seeds here
        model_output[seed_location] = tracker_NN.model[seed_location].predict(model_input[seed_location])
        ## package output into full prediction, in 32x32 array
        for i_seed in range(n_seeds[seed_location]):
            for ilayer in range(tracker_NN.detector.n_layers):
                for ihitstraw in range(tracker_NN.nstraws_perlayer):
                    istraw = model_input[seed_location][i_seed][ilayer][ihitstraw].astype(int)-1
                    if istraw>0:
                        model_prediction[seed_location][i_seed][ilayer][istraw] = model_output[seed_location][i_seed][ilayer][ihitstraw+1]
                    
    n_seeds_total=n_seeds['front']+n_seeds['middle']+n_seeds['back']
    if n_seeds_total==0: 
        print('\nNo seeds in model input' )
        return model_input, np.nan, 0, 0
    ## assign an ID to each seed, so that we can combine and delete them
    seed_ids=np.arange(n_seeds_total)    

    ## Selection based on prediction
    selection_threshold=0.7
    ### Stack all seed locations into single array, for ease of seed counting and id'ing
    model_prediction_full = np.concatenate( (model_prediction['front'],model_prediction['middle'],model_prediction['back']) )
    model_input_full = np.concatenate( (model_input['front'],model_input['middle'],model_input['back']) )
    model_prediction_selected = np.zeros_like(model_prediction_full)
    model_prediction_selected[np.where( model_prediction_full>selection_threshold )] = 1
    
    return model_prediction_selected
    


def get_prediction_ids(tracker_NN, model_prediction_selected):
    ## Get model prediction ids in a single array of dimensions [length,width]
    prediction_ids = np.zeros((tracker_NN.detector.n_layers,tracker_NN.detector.n_straws))
    for i_track in range(model_prediction_selected.shape[0]):
        layers,straws = np.where(model_prediction_selected[i_track]==1)
        prediction_ids[layers,straws] = i_track+1        
    return prediction_ids
    


### Disambiguate hits that are selected by multiple tracks


def disambiguate_hits(tracker_NN, model_input, good_seed_ids,
                      selection_threshold=0.7, drop_ambiguous=True, verbose=False):   
    if verbose: print('\n\n Disambiguating hits\n')
        
    n_seeds,model_output,model_prediction={},{},{}
    for seed_location in ['front','middle','back']:
        n_seeds[seed_location]=model_input[seed_location].shape[0]
        model_prediction[seed_location]=np.zeros( (n_seeds[seed_location],
                                                   tracker_NN.detector.n_layers, tracker_NN.detector.n_straws))
        if n_seeds[seed_location]==0: continue
        ## get model output for all seeds here
        model_output[seed_location] = tracker_NN.model[seed_location].predict(model_input[seed_location])
        ## package output into full prediction, in 32x32 array
        for i_seed in range(n_seeds[seed_location]):
            for ilayer in range(tracker_NN.detector.n_layers):
                for ihitstraw in range(tracker_NN.nstraws_perlayer):
                    istraw = model_input[seed_location][i_seed][ilayer][ihitstraw].astype(int)-1
                    model_prediction[seed_location][i_seed][ilayer][istraw] = model_output[seed_location][i_seed][ilayer][ihitstraw+1]
                    
    n_seeds_total=n_seeds['front']+n_seeds['middle']+n_seeds['back']
    if n_seeds_total==0: # no disambiguation needed
        print('\nNo seeds in model input')
        return model_input, np.nan, 0, 0
    ## assign an ID to each seed, so that we can combine and delete them
    seed_ids=np.arange(n_seeds_total)    

    ### Stack all seed locations into single array, for ease of seed counting and id'ing
    model_prediction_full = np.concatenate( (model_prediction['front'],model_prediction['middle'],model_prediction['back']) )
    model_input_full = np.concatenate( (model_input['front'],model_input['middle'],model_input['back']) )
    model_prediction_selected = np.zeros_like(model_prediction_full)
    model_prediction_selected[np.where( model_prediction_full>selection_threshold )] = 1
    
    ### Go through all good seeds, search for doubly assigned hits
    for seed_id in good_seed_ids:
        if verbose:
            print('processing good seed id #', seed_id)
            print('model input: ', model_input_full[seed_id])
            print('model prediction at : ', np.where(model_prediction_full[seed_id]>0.5))
            print('model prediction selected at: ', np.where( model_prediction_selected[seed_id]>0 ))

        predicted_layers,predicted_straws=np.where(model_prediction_selected[seed_id]==1)
        n_predicted_hits = len(predicted_layers)
        for i,ilayer in enumerate(predicted_layers):
            istraw = predicted_straws[i]
            hit_output_value = model_prediction_full[seed_id][ilayer][istraw]
            if verbose: print(ilayer,istraw, hit_output_value)
            ## Go through other seeds, check for other seeds that selected this hit
            for seed_id2 in good_seed_ids:
                if (seed_id2 == seed_id): continue
                hit_output_value2 = model_prediction_full[seed_id2][ilayer][istraw]
                if hit_output_value2<0.1: continue
                if drop_ambiguous:
                    if verbose: print('dropping ambiguous hit from both seeds')
                    ihit1 = np.where(model_input_full[seed_id][ilayer]==istraw+1)
                    model_input_full[seed_id][ilayer][ihit1] = 0
                    ihit2 = np.where(model_input_full[seed_id2][ilayer]==istraw+1)
                    model_input_full[seed_id2][ilayer][ihit2] = 0
                else:
                    if hit_output_value2>hit_output_value:
                        if verbose: print('changing to new seed : ', hit_output_value2)
                        ihit = np.where(model_input_full[seed_id][ilayer]==istraw+1)
                        model_input_full[seed_id][ilayer][ihit] = 0
                    else: 
                        if verbose: print('staying with same track, but good try: ', hit_output_value2)
                        ihit = np.where(model_input_full[seed_id2][ilayer]==istraw+1)
                        model_input_full[seed_id2][ilayer][ihit] = 0
                    if verbose: print('Disambiguating hit (%d,%d) between seeds %d-%d' % (ilayer,istraw,seed_id,seed_id2) )

                       
    ## Recover partial inputs from each seed location
    model_input['front'] = model_input_full[:n_seeds['front']]
    model_input['middle'] = model_input_full[n_seeds['front']:n_seeds['front']+n_seeds['middle']]
    model_input['back'] = model_input_full[n_seeds['front']+n_seeds['middle']:]

    return model_input





def merge_seeds(tracker_NN, evt_hits, verbose=False):
    '''
    Iteratively go over input for all seeds, merging them into tracks
    and disambiguating difficult hits
    '''
    
    ## Format model input for event
    model_input = tracker_NN.prepare_event_for_prediction(evt_hits)
    n_seeds_total = model_input['front'].shape[0]+model_input['middle'].shape[0]+model_input['back'].shape[0]

    '''
    Ideas to consider:

    * Extract directionality of each track candidate and seeds.
        ** Can use that information to exclude some hit outliers, changes in track direction, or veto some seed combinations.
        ** OTOH, this is too "algorithmic". That information should in principle have been "learnt" by the NN.

    * First identify very good tracks
        ** They will have a high fraction of high-probability hits, and low variance in directionality
            *** We need this metric to be 100% "golden".
            *** If not (eg not fully accurate tracks still get golden output) then make loss function stricter
        ** Merge with other seeds that are included in the golden hits
            *** May mark those other seeds as seeds in the golden track
            *** Then run the track again through the NN, perhaps can gain something with more constraints
            *** Though the NN hasn't been trained to run with seeds in many modules
        ** Then remove the hits of good tracks from all other inputs, making all other tracks much more tractable
        ** Then pass everything through the NN in a second (and more simplified) pass, get better results

    * Then start merging seeds between non-golden tracks
        ** Require a certain number/fraction of shared hits
        ** This might get messy, considering that late seeds have poor output
            *** If the output of a seed looks bad at this point (under some criteria) then just drop it.
            *** Its hits are probably contained in the track of an earlier seed anyway

    * Lastly, disambiguate hits that are shared between tracks
        ** Based on the NN outputs
        
    '''
    looseness = 0
    removed_seed_ids=[]
    good_track_ids=[]
    while (looseness<3):
        ## iterate with increasing looseness

        ## Look for golden tracks, merge their seeds, remove their hits from other inputs
        ## This should iterative, until no new golden tracks found, or any modification in input made
        while(1):
            modified_input=0
            model_input, removed_seed_ids, good_track_ids, modified_input = merge_golden_seeds(tracker_NN, model_input,
                                                                               removed_seed_ids=removed_seed_ids,
                                                                               good_track_ids=good_track_ids, verbose=verbose)
            if ((modified_input==0) or (removed_seed_ids is np.nan) ): break
        ### Are we done?
        if removed_seed_ids is np.nan:
            if verbose: print('Hold up, only 1 or 0 seeds in event')
            return 0, 0
        if (n_seeds_total-len(removed_seed_ids) == len(good_track_ids) ):
            #print('Found all good tracks !!')
            return model_input, good_track_ids


        ## Now look for silver tracks
        ## After each "silver" search we go back to looking for gold
        model_input, removed_seed_ids, good_track_ids, modified_input = merge_silver_seeds(tracker_NN, model_input, looseness=looseness,
                                                                           removed_seed_ids=removed_seed_ids, 
                                                                           good_track_ids=good_track_ids, verbose=verbose)
        if removed_seed_ids is np.nan:
            if verbose: print('Hold up, only 1 or 0 seeds in event')
            return model_input, 0
        if (n_seeds_total-len(removed_seed_ids) == len(good_track_ids) ):
            if verbose: print('Found all good tracks!!')
            return model_input, good_track_ids

        ## loosen criteria for next iteration
        looseness+=1

    ## if we make it out here without final solution, then just return the good tracks we found already
    return model_input, good_track_ids



## here duplicate functionality of fully predicting an event,
## including disambiguating the NN output iteratively

def predict_event(tracker_NN, evt_hits, verbose=False):
    """
    Note that input and prediction are expected to be both for a single event, 
    potentially with many tracks and seeds.
    The first dimension is equal to the number of seeds in the event. 
    * Set all straws that weren't hit to 0
    * Check whether some seeds can be combined into a single track
    * Pass the combined seeds through a (new?) track search
    * Disambiguate straw hits that are assigned to multiple tracks
    * Round probabilistic track assignment to binary yes/no
    * Return the prediction for tracks in the event
    Note that the return prediction matrix will be of smaller dimension than the input
    """ 

    ### Create model input, iteratively merge seeds
    model_input, good_seed_ids = merge_seeds(tracker_NN, evt_hits, verbose=verbose)
    if (np.all(model_input==0)) or (len(good_seed_ids)==0): return np.zeros_like(evt_hits)
    if verbose: 
        print('model input after merge_seeds : ')
        for i in range(model_input['middle'].shape[0]):
            print('Seed #%d, hits: '% (i), np.where(model_input['middle'][i]>0))
        print('good seed_ids : ', good_seed_ids)


    ### Lastly, disambiguate hits that are selected in multiple tracks
    model_input = disambiguate_hits(tracker_NN, model_input=model_input, good_seed_ids=good_seed_ids,
                                    selection_threshold = 0.9, verbose=verbose)
    #if verbose: print('model input after disambiguate_hits : ')
    #for i in range(model_input['middle'].shape[0]):
    #    print('Seed #%d, hits: '% (i), np.where(model_input['middle'][i]>0))


    ### Should be done here!
    ### Get final prediction and selection
    model_prediction_selected = get_prediction_selected(tracker_NN, model_input=model_input,
                                                        selection_threshold=0.8) 

    
    ### Chop down to just the good seeds
    model_prediction_selected = model_prediction_selected[good_seed_ids,:,:]    
    
    if verbose:
        for i in range(model_prediction_selected.shape[0]):
            print('Good seed #%d, selected hits: '% (i), np.where(model_prediction_selected[i]>0))

    prediction_ids = get_prediction_ids(tracker_NN, model_prediction_selected)

    return prediction_ids




def predict_events(tracker_NN, evts_hits, verbose=0):
    """
    Fully predict many events, by calling predict_event on each.
    Return a predictions_ids array of dimensions (n_evts, n_layers, n_straws)
    """ 

    predictions_ids=np.zeros_like(evts_hits)
    n_evts=evts_hits.shape[0]   
    n_print=int(n_evts/10.)
    for i_evt in range(n_evts):
        if verbose and (i_evt%n_print)==0: print('predicting event #%d'%i_evt)
        evt_hits=evts_hits[i_evt]
        predictions_ids[i_evt] = predict_event(tracker_NN, evt_hits, verbose=verbose)
    return predictions_ids


      
def plot_event_and_prediction(tracker_NN, evt_hits, evt_ids=False, verbose=False, figsize=(15,10)):      
    if evt_ids is False:
        print('Event hits:')
        display_event(tracker_NN.detector, evt_hits)
    else:
        print('Event truth:')
        display_event(tracker_NN.detector, evt_hits, evt_ids)

    ## Get model prediction
    prediction_ids = predict_event(tracker_NN, evt_hits, verbose=verbose)
    if np.all(prediction_ids==0): 
        print('No tracks found\n Skipping...')
        return        
    print('\nModel prediction:')    
    
    full_detector_width = tracker_NN.detector.width

    fig, axs = plt.subplots(figsize=figsize)        
    ## First, plot the model input in the left subplot
    axs.set_title('Model prediction')

    axs.set_xlim((-0.1*tracker_NN.detector.length, 1.1*tracker_NN.detector.length))
    axs.set_ylim((-0.1*full_detector_width, 1.1*full_detector_width))

    # plot a circle at the location of each straw
    for ilayer,straw_x in enumerate(tracker_NN.detector.x):
        for straw_y in tracker_NN.detector.y[ilayer]:
            circle = plt.Circle((straw_x, straw_y), tracker_NN.detector.straw_radius, color='black', fill=False)
            axs.add_patch(circle)

    # plot the model prediction for each track
    track_ids=np.unique(prediction_ids[np.where(prediction_ids>0)])
    for i_track in track_ids:              
        track_x,track_y=np.where(prediction_ids==i_track)#self.detector.n_straws]==1)
        axs.scatter(tracker_NN.detector.x[track_x],tracker_NN.detector.y[track_x,track_y], marker='o', #color='black',
                    label='track %d'%(i_track))            
            
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()



