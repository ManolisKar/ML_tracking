#!/usr/bin/python

import os, sys
from ROOT import TFile, TTree
import numpy as np


#import ntups. set up 
f = TFile.Open("/gm2/app/users/ahibbert/MC1/offline/gm2Dev_v9_53_00/run/test/syntheticDataTest/ntups/syntheticDataNtup_test1_station12.root")
tree = f.Get("syntheticData/tree")
N_events = tree.GetEntries()
print('Entries in tree: %d' % N_events)


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


evts_hits = np.zeros((N_events,32,32))
evts_ids  = np.zeros((N_events,32,32))

#loop over events
i_event=0
for event in tree:
    fillArrays(event,evts_hits[i_event],evts_ids[i_event])
    i_event+=1
    

import cPickle as pickle
outfile = open('synthetic_events.pkl','wb')
pickle.dump((evts_hits,evts_ids),
            outfile)
outfile.close()
