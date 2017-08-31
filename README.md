# Earthquake-Early-Warning
Improving Earthquake early warning (EEW) system performance with machine learning and convolutional neural network.

The goal of an EEW system is to rapidly predict earthquake magnitudes using just a few seconds of ground motion time series, such that warnings can be issued before strong ground motion arrives at densely populated areas. Currently deployed EEW systems around the world, relying on several empirical measurements, suffer from either high false positive rate or underestimating magnitudes of large earthquakes. In this project, machine learning techniques are utilized to build a robust, accurate earthuqake magnitude predictor. 

Seismic waveform data have been complied and cleaned for ~10,000 earthquakes over the past three decades in southern California. More than 2000 features have been carefully extracted and engineered from the raw seismic data. Machine learning models including linear regression and random forest reduce the mean squared error by roughly 50% compared to the baseline, one of the EEW models under testing in California. In addtion, a convolutional neural network is trained with the raw data as input, and achieves a similar performance. The random forest model slightly outperforms the other models. More importantly, the baseline underestimates the large earthquake magnitudes by 0.42 units, equivalent to underestimating ground motion by a factor of 4.3. This has been reduced to 0.13-0.15 magnitude units (a factor of 1.6-1.7 for ground motion). If operating in real time, these new EEW algorithms could provide more reliable earthquake magnitude predictions for the purpose of hazard mitigation.

![alt text](https://user-images.githubusercontent.com/28737912/29937854-e55783be-8e54-11e7-9ee7-021c398884d2.png)

Figure 1. The mapview of seismic station (red triangles) and earthquake (blue dots) distribution.

![alt text](https://user-images.githubusercontent.com/28737912/29939745-e8f58542-8e5a-11e7-9936-d5c1d7cde7c7.png) 

Figure 2. Prediciton error distribution for large earthquakes (> M4.5) using ridge regression, random forest and CNN compared to the baseline. The baseline model significantly underestimates the magnitudes of these earthquake. 
