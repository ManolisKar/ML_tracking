# LSTM-based track finding for the Muon g-2 experiment

The Muon g-2 tracker detector consists of 32 layers of thin, long tubes ("straws") filled with gas, which detect the passing of ionizing particles.  
"Track finding" refers to the task of associating straw hits (within a time window of 100 ns) with a specific particle track.  
In the image below the empty circles denote the vertical center of the straw elements, and the black circles indicate the straws that recorded a signal, either from an actual particle or from noise/crosstalk. 

<p align = "center">
<img src="https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/images/window_hits.png?raw=true" alt="Trulli" style="width:100%">
</p>
<p align = "center">
A synthesized tracker event, consisting of 2 crossing tracks and noise hits. The task of track finding is to associate hits with the correct particle track while minimizing errors.
</p>


## The model 

The model structure extends an original idea explored by the [HEP.TrkX project](https://heptrkx.github.io) in [this paper](https://www.epj-conferences.org/articles/epjconf/abs/2017/19/epjconf_ctdw2017_00003/epjconf_ctdw2017_00003.html) and elsewhere.  
In this problem formulation the detector layers are treated as successive "time steps". This motivates the use of a Recurrent Neural Network (RNN) with LSTM layers which are well suited for time-sequence problems. The LSTM layers are followed by a time-distributed fully-connected layer. The output of the NN targets the hits in each detector layer that are associated with the searched track. 

![LSTM_HepTrkX](https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/images/HEP.TrkX_LSTM.png?raw=true)


In our modification of this approach, each input is associated with a "seed": a collection of 3-4 hits in successive layers that appears to come from the same particle track. Note that often there are multiple seeds per track.  
The output of the model then aims to associate hits that are likely to arise from the same particle track as the seed.  
In each layer, the available hits are (_almost like_) one-hot encoded, so that the problem is turned into a classification task for each layer. By considering a reduced number of potential straws (mainly those that are hit, plus some extra positions to maintain constant input dimension) we circumvent the sparsity of the problem to train efficiently and reach 97-98% accuracy on a validation set.  
A further benefit of this formulation is that the fractional output of the model is interpreted as the _confidence_ that a hit belongs with the searched track. In the image below, the colored model output indicates both high confidence for some hits, and some ambiguity at the location where two tracks intersect.


<p align = "center">
<img src="https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/images/model_output.png?raw=true" alt="Trulli" style="width:100%">
</p>
<p align = "center">
<sup><sub>
Model output, for the input using the seed circled in the left. The output for each hit is interpreted as the probability that it belongs with the same track as the seed.
</sub></sup>
</p>

The last part of the algorithm is an iterative procedure to merge different seeds into the same track, and to resolve ambiguities and uncertainties. This procedure makes use of both:
* high-confidence information, eg when a track is found, its hits are removed from the input of other seeds that are still facing ambiguity, to make their task easier; and
* low-confidence information, where we may decide to drop ambiguous hits from consideration to avoid errors and make it easier to converge to a track.  

This updating to use all available information and dealing with uncertainty is reminiscent of a Kalman filter, an algorithm that has also been traditionally used in tracking applications.  

> This hybrid approach, consisting of a recursive NN followed by an algorithmic step where we process the model's output iteratively, results in an algorithm that is more interpretable, and in which we have more control to tune for the desired performance parameters.



## A smarter algorithm

We developed a [synthetic dataset](https://github.com/ManolisKar/ML_tracking/tree/main/MC) for training, performance evaluation and comparison with the main (currently used) tracking algorithm. 
In the image below you see the comparison in track finding performance between the existing tracking algorithm ("Main Tracking") and our RNN model, on the same event we have been examining in this page. This being quite a challenging event with crossing tracks and noise hits, the main tracking algorithm makes several mistakes (marked by Xs in the image) and even breaks a particle track in two. Our model on the other hand is able to powerfully associate hits with seed segments, and our algorithm can drop ambiguous hits without losing the "big picture" of the event.

![comparison](https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/images/comparison.png?raw=true)


Overall our algorithm is proven to be "smarter" than the one currently used in the experiment. It finds track candidates that are much more pure (error-free) and much more likely to be reconstructed successfully. This improved performance can potentially deliver significant gains for the experimenal targets. 



## Performance metrics

Critical to characterize our model's performance is the development of useful metrics. 
We develop several custom metrics to characterize different aspects of this complicated task:
* The fraction of the true track's hits that are correctly identified as part of the track, averaged over the population of found tracks;
* The fraction of assigned hits that are wrongly identified with a track; 
* The total number of tracks found, as a fraction of the true number of tracks in the test dataset. Also quoted is the number of "duplicate" tracks, ie tracks which an algorithm wrongly splits into two or more, as in the image above.

These performance metrics allow us to tune the model for the desired performance. In the image below we plot results from several trials on the merging algorithm. There are 2 points from each trial, a blue one denoting hit assignment accuracy and a red one on the number of tracks found in the trial. They are both plotted on the same vertical location for the hit assignment error for that trial. Admittedly this isn't an easy plot to read, but it is useful to condense information along multiple performance dimensions.  
Also plotted are two arrows that demonstrate necessary trade-offs between the performance metrics. Eg, the red arrow suggests that if we need to find more tracks in the dataset, we need to be more bold with out merging choices, thereby also increasing our error rate. From this set of trials we can select a merging strategy that fits our requirements and which performs better than the average trends. 
A separate but similar tuning of the RNN hyperparameters and architecture was also performed.

![merging_trials](https://github.com/ManolisKar/ML_tracking/blob/main/track_finding/RNN/images/merging_trials.png?raw=true)




The performance of the main tracking and our RNN model along those metrics is compared in the table below.  



|                                                          | Main Tracking       | LSTM-based Model |
|----------------------------------------------------------|      -----:         |-----------:        |
| **Precision on found tracks**                                | 88.4%               | 92.5%            |
| **Wrongly assigned hits**                                    | 5.8%                | 2.8%             |
| **# of found tracks [and split tracks]       (out of 1970)** | 2167 [149]  (110.0%) | 1799 [6]  (91.3%) |




It can be seen that tracks found by our RNN model contain fewer errors and more hits from the true track, on average. The existing ("main tracking") algorithm finds many tracks of low quality, many of which are duplicates or contain errors. The result is that many of these track candidates will go on to fail the reconstruction process, which consists of expensive simulation iterations, straining computational resources. This is a significant consideration for the collaboration and for the Fermilab Scientific Computing Division. Our algorithm on the other hand returns fewer track candidates of significantly higher purity, with many potential benefits:
* Computational resources required for the processing of found tracks are reduced by approximately 17%. 
* The errors in found tracks are slashed in half, potentially yielding more accurate information on the beam dynamics of the muon distribution whence those particles decayed.
* It's possible that with track candidates of higher purity, we can increase the number of found tracks, even when starting with a smaller population of candidates.