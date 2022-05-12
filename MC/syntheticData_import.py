#!/usr/bin/python

import os, sys
from ROOT import TFile, TTree, TChain
import numpy as np
import glob


## Read in data
''' This is a method for single root file
f = TFile.Open("/gm2/app/users/ahibbert/MC1/offline/gm2Dev_v9_53_00/run/test/syntheDataNoiseOn/ntups/syntheticDataNtup_test1_station12.root")
tree = f.Get("syntheticData/tree")
N_events = tree.GetEntries()
print('Entries in tree: %d' % N_events)
'''

## Use below for chain of trees
rootfiledir = '/gm2/app/users/ahibbert/MC1/offline/gm2Dev_v9_53_00/run/test/synthDataNoiseOn/ntups/'
chain = TChain('syntheticData/tree')
nfiles=0
for rootfile in glob.glob(rootfiledir+'/syntheticDataNtup*.root'):
    chain.Add(rootfile)
    nfiles+=1

print('Number of files processed: %d' % nfiles)
N_events = chain.GetEntries()
print('Entries in chain of trees: %d' % N_events)


def fillArrays(ev, evt_hits, evt_ids):
    ## Fill the 32x32 hits and ids arrays for this event    
    layersA = ev.layerNumsA
    strawsA = ev.strawNumsA
    evt_ids[layersA,strawsA]=1
    
    layersB = ev.layerNumsB
    strawsB = ev.strawNumsB
    evt_ids[layersB,strawsB]=2
    
    layersC = ev.layerNumsComb
    strawsC = ev.strawNumsComb
    evt_hits[layersC,strawsC]=1

    
def fillArrays_A(ev, evt_hits, evt_ids):
    ## Fill the 32x32 hits and ids arrays for this event, but only using track A    
    layersA = ev.layerNumsA
    strawsA = ev.strawNumsA
    evt_ids[layersA,strawsA]=1
    evt_hits[layersA,strawsA]=1

def fillArrays_B(ev, evt_hits, evt_ids):
    ## Fill the 32x32 hits and ids arrays for this event, but only using track B
    layersB = ev.layerNumsB
    strawsB = ev.strawNumsB
    evt_ids[layersB,strawsB]=1
    evt_hits[layersB,strawsB]=1


evts_hits = np.zeros((N_events,32,32))
evts_ids  = np.zeros((N_events,32,32))
evts_hits_WithSingles = np.zeros((3*N_events,32,32))
evts_ids_WithSingles  = np.zeros((3*N_events,32,32))

#loop over events
i_event=0
i_event_WS=0
#for event in tree:
for event in chain:
    fillArrays(event,evts_hits[i_event],evts_ids[i_event])
    i_event+=1
    fillArrays(event,evts_hits_WithSingles[i_event_WS],evts_ids_WithSingles[i_event_WS])
    i_event_WS+=1
    fillArrays_A(event,evts_hits_WithSingles[i_event_WS],evts_ids_WithSingles[i_event_WS])
    i_event_WS+=1
    fillArrays_B(event,evts_hits_WithSingles[i_event_WS],evts_ids_WithSingles[i_event_WS])
    i_event_WS+=1
    

import cPickle as pickle
outfile = open('synthetic_events.pkl','wb')
pickle.dump((evts_hits,evts_ids),
            outfile)
outfile.close()
outfile = open('synthetic_events_WithSingles.pkl','wb')
pickle.dump((evts_hits_WithSingles,evts_ids_WithSingles),
            outfile)
outfile.close()
