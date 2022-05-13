# Classifying quality of reconstructed tracks

The Muon g-2 tracking framework identifies series of signals in detector elements that are likely to arise from the passage of the same particle. It then reconstructs the particle **track**, and extrapolates the particle (usually a positron) trajectory back to its origin - the muon decay vertex. Extraction of the vertex coordinates is an important goal of the tracking procedure, as it yields necessary information on the parent muon distribution.

Stringent Quality Cuts (**QC**) are applied on the reconstructed tracks to ensure that errors in the reconstruction process are avoided. Eg the tracks are required to contain a minimum number of detector elements hit by the particle, and meet a minimum reconstruction fit probability, among other selection criteria. It turns out that the QC remove the majority of reconstructed track candidates.

<p align = "center">
<img src="https://github.com/ManolisKar/ML_tracking/blob/main/track_quality/images/resolutions_QC.png?raw=true" alt="Trulli" style="width:40%">
</p>
<p align = "center">
<sup>
Vertex resolution (combined radial and vertical) of a full simulated sample of track candidates (in blue) and those that survive the QC (in orange). The majority of track candidates are removed by the QC. The units of the combined resolution are in $mm$. 
</sup>
</p>


This motivates us to use ML classifiers to try to identify track candidates of high quality more efficiently than the QC.


## The classification task 

The parameters of interest that we will use to define the quality of a track candidate are the resolutions of the extracted vertex coordinates, where we are interested in the radial (R) and vertical (Y) coordinates. 
We define a track's resolution as the distance of the extracted coordinate from its true value in a simulated sample. 

Our approach will be stronger by simplifying the task into a binary classification of "good" or "bad" track candidates. 
The QC are basically a Decision Tree classifier that performs this binary classification based on track parameters. 
To develop new classifiers and train them in a supervised manner, we define the selection criterion as combined resolution of 


## A collection of classifiers