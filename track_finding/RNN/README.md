# LSTM-based track finding for the Muon g-2 experiment

The Muon g-2 tracker detector consists of 32 layers of thin, long tubes ("straws") filled with gas, which detect the passing of ionizing particles.  
"Track finding" refers to the task of associating straw hits (within a time window of 100 ns) with a specific particle track.  
In the image below the empty circles denote the vertical center of the straw elements, and the black circles indicate the straws that recorded a signal, either from an actual particle or from noise/crosstalk. 

![Window hits](https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/images/window_hits.png?raw=true)


## The model 

The model structure extends an original idea explored by the [HEP.TrkX project](https://heptrkx.github.io) in [this paper](https://www.epj-conferences.org/articles/epjconf/abs/2017/19/epjconf_ctdw2017_00003/epjconf_ctdw2017_00003.html) and elsewhere.  
In this problem formulation the detector layers are treated as successive "time steps". This motivates the use of a Recurrent Neural Network (RNN) with LSTM layers which are well suited for time-sequence problems. The LSTM layers are followed by a time-distributed fully-connected layer. The output of the NN targets the hits in each detector layer that are associated with the searched track. 

![LSTM_HepTrkX](https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/images/HEP.TrkX_LSTM.png?raw=true)

In our modification of this approach, each input is associated with a "seed": a collection of 3-4 hits in successive layers that appears to come from the same particle track. Note that often there are multiple seeds per track.  
The output of the model then aims to associate hits that are likely to arise from the same particle track as the seed.  
In each layer, the available hits are (_almost like_) one-hot encoded, so that the problem is turned into a classification task for each layer. Through considering a reduced number of potential straws (mainly those that are hit, plus some extra positions to maintain constant input dimension) we circumvent the sparsity of the problem to train efficiently and reach 97-98% accuracy on a validation set.  
A further benefit of this formulation is that the fractional output of the model is interpreted as the _confidence_ that a hit belongs with the searched track. In the image below, the colored model output indicates both high confidence for some hits, and some ambiguity at the location where two tracks intersect.

![model output](https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/images/model_output.png?raw=true)

The last part of the algorithm is an iterative procedure to merge different seeds into the same track, and to resolve ambiguities and uncertainties. This procedure makes use of both:
* high-confidence information, eg when a track is found, its hits are removed from the input of other seeds that are still facing ambiguity, to make their task easier; and
* low-confidence information, where we may decide to drop ambiguous hits from consideration to avoid errors and make it easier to converge to a track.  

This updating to use all available information and dealing with uncertainty is reminiscent of a Kalman filter, an algorithm that has also been traditionally used in tracking applications.



## Performance

We developed a synthetic dataset for training, performance evaluation and comparison with the main (currently used) tracking algorithm. 
In the image below you see the comparison in track finding performance between the main tracking and our RNN model and algorithm, on the same event we have been examining in this page. This being a quite challenging event with crossing tracks and noise hits, the main tracking algorithm makes several mistakes (marked by Xs in the image) and even breaks a particle track in two. Our model on the other hand is able to powerfully associate hits with seed segments, and our algorithm can drop ambiguous hits without losing the "big picture" of the event.

![comparison](https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/images/comparison.png?raw=true)


Overall our algorithm is proven to be "smarter" than the one currently used in the experiment. It finds track candidates that are much more pure (error-free) and much more likely to be reconstructed successfully. This improved performance can potentially deliver significant gains for the experimenal targets. 