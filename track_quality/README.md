# Classifying quality of reconstructed tracks

The Muon g-2 tracking framework identifies series of signals in detector elements that are likely to arise from the passage of the same particle. It then reconstructs the particle **track**, and extrapolates the particle (usually a positron) trajectory back to its origin - the muon decay vertex. Extraction of the vertex coordinates is an important goal of the tracking procedure, as it yields necessary information on the parent muon distribution.

Stringent Quality Cuts (**QC**) are applied on the reconstructed tracks to ensure that errors in the reconstruction process are avoided. Eg the tracks are required to contain a minimum number of detector elements hit by the particle, and meet a minimum reconstruction fit probability, among other selection criteria. It turns out that the QC remove the majority of reconstructed track candidates.

<p align = "center">
<img src="https://github.com/ManolisKar/ML_tracking/blob/main/track_quality/images/resolutions_QC.png?raw=true" alt="Trulli" style="width:40%">
</p>
<p align = "center">
<sup>
Fig. 1: Vertex resolution (combined radial and vertical) of a full simulated sample of track candidates (in blue) and those that survive the QC (in orange). The majority of track candidates are removed by the QC. The units of the combined resolution are in mm. 
</sup>
</p>


This motivates us to use ML classifiers to try to identify track candidates of high quality more efficiently than the QC.


## The classification task 

The parameters of interest that we will use to define the quality of a track candidate are the resolutions of the extracted vertex coordinates, where we are interested in the radial (R) and vertical (Y) coordinates. 
We define a track's resolution as the distance of the extracted coordinate from its true value in a simulated sample. 

Our approach will be stronger by simplifying the task into a binary classification of "good" or "bad" track candidates. 
The QC is basically a Decision Tree classifier that performs this binary classification based on track parameters. 
To develop new classifiers and train them in a supervised manner, we define the selection criterion as a combined resolution (the quantity plotted in Fig. 1) of 5 mm. Tracks that achieve better (smaller) resolution than this are labeled as "good", otherwise "bad".

Notice that the 5 mm quality threshold is tighter than the actual QC-selected distribution. 
That is good, we want to be conservative in our approach, to justify a classification which selects potentially many more tracks. 


## A collection of classifiers

We try several classifiers: DecisionTree, GradientBoosting, SVM, AdaBoost. We tune the hyperparameters for all of them. 
Finally we extract performance metrics on a test set. The simplest comparative depiction of their performance is shown in Fig. 2, with their Precision-Recall curves. 


<p align = "center">
<img src="https://github.com/ManolisKar/ML_tracking/blob/main/track_quality/images/precision_recall.png?raw=true" alt="Trulli" style="width:50%">
</p>
<p align = "center">
<sup>
Fig. 2: Precision-Recall curves for four tuned classifiers.
</sup>
</p>

It can be seen that the DecisionTree classifier performs worst, so we drop it. 
In Fig. 3 we plot the resolution distributions for tracks selected from each of the three remaining classifiers. 
We also plot results from combining those classifiers, either via "Voting" (demanding that at least two out of the three classifiers select a track as good) or by "All" (demanding that all three classifiers select a track).


<p align = "center">
<img src="https://github.com/ManolisKar/ML_tracking/blob/main/track_quality/images/resolution_classifiers.png?raw=true" alt="Trulli" style="width:100%">
</p>
<p align = "center">
<sup>
Fig. 3: Radial and vertical resolutions of tracks accepted by the three classifiers, and their combinations.
</sup>
</p>


We observe that the three classifiers select very similar distributions of tracks. By using their combinations we can capture the subtle complementary insights between them, while being much more stable against outliers. 
In the table below we list the number of tracks accepted from each classifier choice, along with the selection truth. 
Based on this, a reasonably conservative approach would be to select only tracks accepted by All three classifiers. That would make any remaining outliers practically consistent with QC, while still **more than doubling the number of accepted high-quality tracks**. 

In Fig. 4 we remove individual classifiers to see the resolution comparison more clearly.  It's good to see also that the resulting distributions of accepted tracks have good resolutions in the separate dimensions, even though we used a combined quality criterion. 


| Classifier  |  # accepted tracks | 
| ----------  | -------: |
| AdaBoost    | 221 |
| GradientBoosting| 214 | 
| SVM | 228 | 
| Voting | 223 |
| All | 168 |
| Selection truth | 237 |
| QC | 74 |




<p align = "center">
<img src="https://github.com/ManolisKar/ML_tracking/blob/main/track_quality/images/resolution_final.png?raw=true" alt="Trulli" style="width:100%">
</p>
<p align = "center">
<sup>
Fig. 4: Radial and vertical resolutions of tracks accepted by All three ML classifiers (blue), and by QC (red).
</sup>
</p>


## Assessing and next steps

So we demonstrated that our combined classifiers identify many more high-quality tracks than the QC, and with very good resolution properties.
We still however cannot blindly proceed to implement the new classification scheme (even if we assume that the simulation we used is a good representation of reality). We have to examine the track parameters used in the QC to understand where the differences arise from. 
Some of these track parameters (about half) are plotted in Fig. 5. 


<p align = "center">
<img src="https://github.com/ManolisKar/ML_tracking/blob/main/track_quality/images/track_parameters.png?raw=true" alt="Trulli" style="width:100%">
</p>
<p align = "center">
<sup>
Fig. 5: Distributions of track parameters used in QC, for the tracks selected by QC (red) and by the ML classifier combinations (blue and orange).
</sup>
</p>

I won't go into explaining all of these, as apparently only one is important: "pValue" plotted in the bottom right. 
That is the fit probability of the reconstructed trajectory. Unfortunately it looks like all the extra tracks have very low, near-zero fit probability. 
In fact a feature selection exercise using the code below, reveals that the fit probability is a very poor feature to classify quality tracks, counter-intuitively. 


```
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 ## needs non-negative values, therefore rescaling of input is necessary
from sklearn.feature_selection import mutual_info_classif

## list of potentially useful variables/features
clf_vars=['RecoTrackX','RecoTrackY','RecoTrackZ',
          'pValue', 'nHits', 'nUHits','nVHits', 'MissedLayersFrac','MinDriftTime','MaxDriftTime','MaxResidual',
          'RecoVertexX', 'RecoVertexY', 'RecoVertexZ', 'RecoVertexR',
          'RecoVertexUncP', 'RecoVertexUncY', 'RecoVertexUncPY', 'RecoVertexUncR',
          'RecoVertexUncPR', 'hitVolume', 'RecoExtrapolatedDistance',
          'CaloVertexMomUnc', 'CaloVertexUncX',
          'CaloVertexUncY', 'CaloVertexUncPX', 'CaloVertexUncPY','diff_UV', 'RecoVertexAzimuth']

X=df_cutout[clf_vars].values
y=df_cutout['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#apply SelectKBest class to extract best features
best_features = SelectKBest(score_func=mutual_info_classif, k=10)
fit = best_features.fit(X_train,y_train)

df_features=pd.DataFrame()
df_features['Feature']=clf_vars
df_features['Score']=fit.scores_
df_features.sort_values(by='Score',ascending=False)
```



<p align = "center">
<img src="https://github.com/ManolisKar/ML_tracking/blob/main/track_quality/images/feature_importance.png?raw=true" alt="Trulli" style="width:10%">
</p>
<p align = "center">
<sup>
Fig. 6: Feature importance scores of several QC parameters. The reconstruction fit probability "pValue" is counter-intuitively very poor at predicting tracks with good resolution. 
</sup>
</p>


So how is it possible that tracks with very low fit probability achieve excellent vertex resolution? 
Something that could be happening is that the track candidates contain a hit that didn't actually come from the same particle, but is instead arising from noise or crosstalk, or just from another particle track. 
Then the outlying hit would destroy the fit probability, while not significantly altering the vertex reconstruction. 

If that is the case, then identifying this large population of tracks with low fit probability but very good vertex resolution could be of massive benefit. 
Those tracks could be treated for outliers and re-examined, potentially doubling the number of high-quality tracks. 
For us however, this was a signal that we needed to work further upstream in the tracking chain to improve the quality of the incoming track candidates. 
This is what motivated our work on the [LSTM-based track-finding algorithm](https://github.com/ManolisKar/ML_tracking/tree/main/track_finding/RNN). 
