import numpy as np

def remove_nearby(hit_straws, unique_within=2):
    '''
    Remove all hits in list that are closer together than allowed
    unique_within=4 #in each layer the hit must be unique within +-4 straws
    '''
    hit_straws=np.array(hit_straws)
    for straw in hit_straws:
        other_straws=[i for i in hit_straws if i!=straw]
        if (abs(other_straws-straw)<=unique_within).any():
            ## found hits nearby, remove them
            hit_straws=hit_straws[np.where( abs(hit_straws-straw)>unique_within)]
    return hit_straws

def get_next_layer_candidates(layer_hits, previous_layer_hit, max_diff=2):
    '''
    Get all candidate hits (should be 1 at most, as this should be applied after nearby hits are removed)
    that could physically belong to the same track as the hit in the previous layer,
    by checking that they are within the allowed max-diff.
    Returned list should have a length of 0 or 1.
    '''
    layer_hits=[hit for hit in layer_hits if abs(hit-previous_layer_hit)<=max_diff]
    return layer_hits

def accept_seed_candidate(candidate_hits,rms_cutoff=1):
    '''
    Accept or reject a seed candidate based on its RMS
    *** For now this is not implemented, the max-diff requirement should be enough ***
    '''
    if np.std(candidate_hits)<rms_cutoff: return True
    else: return False


def get_first_layer_seeds(ix,iy,first_layer):
    """
    Get all seeds that have a hit in the first layer of a module.
    Parameters:
        ix,iy: x,y coordinates of all hits in the module
        first layer: id of 1st layer in module
    Returns:
        x,y coordinates for all hits in a seed, OR empty if no seeds found.
        the ix,iy but will all hits that have been entered in seeds removed.
    """
    
    seeds_xy=[]
    candidate_straw_0,candidate_straw_1,candidate_straw_2,candidate_straw_3=0,0,0,0

    candidate_straws_0 = iy[ np.where(ix==first_layer) ]
    candidate_straws_0=remove_nearby(candidate_straws_0)
    if len(candidate_straws_0)==0: return seeds_xy, (ix,iy)

    for candidate_straw_0 in candidate_straws_0:
        ## we have a hit from 1st layer, go on to 2nd layer
        candidate_straws_1 = iy[ np.where(ix==first_layer+1) ]
        candidate_straws_1 = remove_nearby(candidate_straws_1)
        candidate_straws_1 = get_next_layer_candidates(candidate_straws_1,candidate_straw_0)
        if len(candidate_straws_1)==0: 
            ## layer-1 missed, go on to layer-2
            candidate_straws_2 = iy[ np.where(ix==first_layer+2) ]
            candidate_straws_2 = remove_nearby(candidate_straws_2)
            candidate_straws_2 = get_next_layer_candidates(candidate_straws_2,candidate_straw_0)
            if len(candidate_straws_2)==0: 
                ## second layer without hit, drop candidate
                continue
            candidate_straw_2 = candidate_straws_2[0]
            ## hits in layers 0,2, move to layer 3
            candidate_straws_3 = iy[ np.where(ix==first_layer+3) ]
            candidate_straws_3 = remove_nearby(candidate_straws_3)
            candidate_straws_3 = get_next_layer_candidates(candidate_straws_3,candidate_straw_2)            
            if len(candidate_straws_3)==0: 
                ## second layer without hit, drop candidate
                continue
            candidate_straw_3 = candidate_straws_3[0]
            
            #if not accept_seed_candidate(candidate_y): continue
            ## accept seed candidate
            candidate_x=[first_layer, first_layer+2, first_layer+3]
            candidate_y=[candidate_straw_0, candidate_straw_2, candidate_straw_3]
            seed_xy = (candidate_x,candidate_y)
            seeds_xy+=[seed_xy]

        #if len(candidate_straws_1)==0:
        else:
            ## continue with layers 0,1 hit
            candidate_straw_1 = candidate_straws_1[0]
            ## go on to layer-2
            candidate_straws_2 = iy[ np.where(ix==first_layer+2) ]
            candidate_straws_2 = remove_nearby(candidate_straws_2)
            candidate_straws_2 = get_next_layer_candidates(candidate_straws_2,candidate_straw_0)
            if len(candidate_straws_2)==0: 
                ## layer-2 missed, go on to layer-3
                candidate_straws_3 = iy[ np.where(ix==first_layer+3) ]
                candidate_straws_3 = remove_nearby(candidate_straws_3)
                candidate_straws_3 = get_next_layer_candidates(candidate_straws_3,candidate_straw_1)            
                if len(candidate_straws_3)==0: 
                    ## second layer without hit, drop candidate
                    continue
                candidate_straw_3 = candidate_straws_3[0]

                #if not accept_seed_candidate(candidate_y): continue
                ## accept seed candidate
                candidate_x=[first_layer, first_layer+1, first_layer+3]
                candidate_y=[candidate_straw_0, candidate_straw_1, candidate_straw_3]
                seed_xy = (candidate_x,candidate_y)
                seeds_xy+=[seed_xy]

            #if len(candidate_straws_2)==0: 
            else:
                ## continue with layers 0,1,2 hit            
                candidate_straw_2 = candidate_straws_2[0]
                ## go on to layer-3
                candidate_straws_3 = iy[ np.where(ix==first_layer+3) ]
                candidate_straws_3 = remove_nearby(candidate_straws_3)
                candidate_straws_3 = get_next_layer_candidates(candidate_straws_3,candidate_straw_2)            
                if len(candidate_straws_3)==0:
                    ## we missed the last layer, but we are accepting the seed anyway from the previous 3 layers
                    #if not accept_seed_candidate(candidate_y): continue
                    ## accept seed candidate
                    candidate_x=[first_layer, first_layer+1, first_layer+2]
                    candidate_y=[candidate_straw_0, candidate_straw_1, candidate_straw_2] 
                    seed_xy = (candidate_x,candidate_y)
                    seeds_xy+=[seed_xy]
                else:
                    candidate_straw_3 = candidate_straws_3[0]
                    #if not accept_seed_candidate(candidate_y): continue
                    ## accept seed candidate
                    candidate_x=[first_layer, first_layer+1, first_layer+2, first_layer+3]
                    candidate_y=[candidate_straw_0, candidate_straw_1, candidate_straw_2, candidate_straw_3] 
                    seed_xy = (candidate_x,candidate_y)
                    seeds_xy+=[seed_xy]

    ## finished processing module, return found seeds
    ## also return hit coordinates, after removing hits in seeds (so they can be entered in the 2nd-layer seed finder)
    for seed_xy in seeds_xy:
        seed_x,seed_y=seed_xy
        for iseed,iseed_x in enumerate(seed_x):
            iseed_y=seed_y[iseed]
            for ihit,ihit_x in enumerate(ix):
                ihit_y=iy[ihit]
                if (ihit_x==iseed_x) & (ihit_y==iseed_y):
                    ix=np.delete(ix,ihit)
                    iy=np.delete(iy,ihit)
                    break
    remaining_xy=(ix,iy)
    return seeds_xy, remaining_xy



def get_second_layer_seeds(ix,iy,first_layer):
    """
    Get all seeds that have a hit in the second layer of a module.
    Assumes that all hits from seeds found beginning in 1st layer have been removed.
    Parameters:
        ix,iy: x,y coordinates of **remaining** hits in the module
        first layer: id of 1st layer in module
    Returns:
        x,y coordinates for all hits in a seed, OR empty if no seeds found.
    """
    
    seeds_xy=[]
    candidate_straw_1,candidate_straw_2,candidate_straw_3=0,0,0

    candidate_straws_1 = iy[ np.where(ix==first_layer+1) ]
    candidate_straws_1=remove_nearby(candidate_straws_1)
    if len(candidate_straws_1)==0: return seeds_xy

    for candidate_straw_1 in candidate_straws_1:
        ## we have a hit from 2nd layer, go on to 3rd layer
        candidate_straws_2 = iy[ np.where(ix==first_layer+2) ]
        candidate_straws_2 = remove_nearby(candidate_straws_2)
        candidate_straws_2 = get_next_layer_candidates(candidate_straws_2,candidate_straw_1)
        if len(candidate_straws_2)==0: return seeds_xy
        candidate_straw_2 = candidate_straws_2[0]
        ## go on to layer-3
        candidate_straws_3 = iy[ np.where(ix==first_layer+3) ]
        candidate_straws_3 = remove_nearby(candidate_straws_3)
        candidate_straws_3 = get_next_layer_candidates(candidate_straws_3,candidate_straw_2)
        if len(candidate_straws_3)==0: return seeds_xy
        ## layers 1,2,3 hit, accept candidate
        candidate_straw_3 = candidate_straws_3[0]
        #if not accept_seed_candidate(candidate_y): continue
        candidate_x=[first_layer+1, first_layer+2, first_layer+3]
        candidate_y=[candidate_straw_1, candidate_straw_2, candidate_straw_3] 
        seed_xy = (candidate_x,candidate_y)
        seeds_xy+=[seed_xy]

    ## finished processing module, return found seeds
    return seeds_xy



def make_real_seeds(evt_hits):
    """
    Get all pure-appearing seeds within a time window.
    Makes no use of truth information.
    Parameters:
        evt_hits: 2D array of seeds within a window
    Returns:
        An array with x,y coordinates of all seeds.
        A 2D array (same dimension as hits) with all seed hits.
        
    """

    ## For now assume as hard-coded (this can be easily amended by passing the detector)
    n_modules=8 ## number of modules, each with 4 layers of straws
    n_layersPerModule=4
        
    seeds_xy=[]
    evt_seeds = np.zeros_like(evt_hits)
    
    ## layer/straw for all hits in event
    hit_x,hit_y = np.where(evt_hits>0)
    
    ## iterate over all modules, find seeds that fulfill requirements
    for i_module in range(n_modules):
        first_layer = n_layersPerModule*i_module
        # get hits on this module
        mask = (hit_x>=first_layer) & (hit_x<first_layer+n_layersPerModule)
        ix,iy= hit_x[mask], hit_y[mask]
        if len(ix)<3: continue
        # get seeds that start in 1st layer of this module
        module_seeds_xy, remaining_hits = get_first_layer_seeds(ix,iy,first_layer)
        if len(module_seeds_xy)>0:
            seeds_xy+=module_seeds_xy
            for module_seed_xy in module_seeds_xy:
                seed_x,seed_y=module_seed_xy
                evt_seeds[(seed_x,seed_y)]=1
        # then get seeds starting in the 2nd layer
        remaining_x,remaining_y = remaining_hits
        module_seeds_xy = get_second_layer_seeds(remaining_x,remaining_y,first_layer)
        if len(module_seeds_xy)>0:
            seeds_xy+=module_seeds_xy
            for module_seed_xy in module_seeds_xy:
                seed_x,seed_y=module_seed_xy
                evt_seeds[(seed_x,seed_y)]=1

    return seeds_xy, evt_seeds


