# Data augmentation for RNN model training

The [LSTM-based model used for track finding](https://github.com/ManolisKar/ML_tracking/tree/main/track_finding/RNN) can be trained in a supervised manner to perform its pattern-finding task. 
In this page we describe the pseudo- and synthetic datasets created for training the model.


## Pseudo-data

As a simple first approach we generate fully simulated pseudo-data to train the model. The simulated data do not aim to reproduce reality to a high degree, but rather provide a very useful training ground to develop the track-finding algorithm and provide proof of principle that this complicated novel approach can succeed. 

The Detector class in [detector.py](https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/src/detector.py) has a method to create a 2-dimensional approximation of the tracker detector elements (straws), while also defining parameters such as occupancy. Then we can generate simulated tracks with randomized trajectory parameters on that 2-dimensional plane, and identify straw "hits" when the trajectory intersects them.  

We experimented with several degrees of freedom, such as occupancy, to characterize the track finding algorithm. In the image below we see the model's prediction for a complicated pseudo-event with 5 tracks, much higher occupancy than actually encountered in the experiment. While this is an early version of the track-finding model and algorithm, it exhibits powerful pattern-finding capabilities, assigning nearly all hits with the correct track. 

![pseudoevent](https://github.com/ManolisKar/ML_tracking/blob/main/MC/images/pseudoevent.png?raw=true)


Note that the pseudo-world we are generating is 2-dimensional, when in reality both the straws and the tracks have 3-dimensional extent and parameters. 
However this is not an obstacle for the framing and development of the algorithm, as the model represents each straw with a single element anyway. 
Therefore this simulation achieves the required data input representation and facilitated much of the model and algorithm development. 
The same model architecture can be applied on a real event, but of course its performance would be reduced as the training set is not fully realistic. 
It is missing effects from the full 3-dimensional geometry, from the experiment's magnetic field, from any straw misalignments, etc.  
Therefore we built a synthetic dataset that captures all these elements. A model trained on that dataset will update its *internal representation* of the connections between straw hits to achieve the track finding supervised task.





## Synthetic dataset

Some of the elements missing from the simple simulation, such as effects from the varying magnetic field and subtle straw misalignments, are either unknown or expensive to implement in a full simulation. They are only captured in real experimental data, so that is where we turn to.  

We can isolate real particle tracks with little to no ambiguity from late in the muon fill, when event rate is very low. These tracks are a high-purity sample that capture all the effects mentioned before that are difficult to implement in simulation. 
From that high-purity sample, we overlay multiple tracks together in synthetic events, like the one shown in the image below. Approximations of crosstalk and noise effects can be artificially turned on as well, making the synthetic events practically indistinguishable from real ones.  


![synthetic_event](https://github.com/ManolisKar/ML_tracking/blob/main/MC/images/synthetic_event.png?raw=true)


> A synthetic dataset thus created includes target labels for supervised training. By training our model to identify the real tracks in this dataset, we ensure it will be highly applicable on real unlabeled production data as well.


If there are any concerns on the artificiality of such process, for example on the realism of the artificial noise and crosstalk hits, then a *semi-synthetic* dataset can be developed. 
In this implementation, the high-purity isolated tracks can be overlaid on real production events, that contain otherwise fully unlabeled hits. 
Since the input to the model is at the seed level, we do not require full knowledge of the entire event, just of the target track associated with that seed. 
Therefore we can train our model on the labeled tracks, against a background of fully real tracks, as well as any noise and crosstalk. 
A model thus trained should be highly applicable for production data. Furthermore on this semi-synthetic dataset we can perform a powerful characterization of track-finding efficiency versus time in the muon fill, for competing algorithms.  