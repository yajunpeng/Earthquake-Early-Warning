# Earthquake-Early-Warning
Improving Earthquake early warning (EEW) system performance with machine learning and convolutional neural network.

The goal of an EEW system is to rapidly predict earthquake magnitudes, such that warnings can be issued before strong ground motion arrives. Currently deployed EEW systems around the world, relying on several empirical measurements, suffer from either high false positive rate or underestimating of large earthquakes. In this project, machine learning techniques are utilized to build a better earthuqake magnitude predictor. 

Ground motion time series data have been complied and cleaned for ~10,000 earthquakes over the past three decades in southern California. All data are available for downloading at the IRIS seismic data center.
![alt text](https://user-images.githubusercontent.com/28737912/29937854-e55783be-8e54-11e7-9ee7-021c398884d2.png) Figure 1. The mapview of seismic station (red triangles) and earthquake (blue dots) distribution.

After feature extraction from raw seismic waveform data, the machine learning models including linear regression and random forest reduce the mean squared error by 50%, compared to the baseline model that depends on solely on the peak ground displacement. In addtion, a convolutional neural network is trained with the raw data as input, which achieves a similar performance. The random forest model slightly outperforms the other models. More importantly, the issue of underestimating large earthquake magnitudes is significantly alleviated. If operated in real time, this new EEW system could potentially provide reliable and accurate earthquake magnitude predictions for the purpose of hazard mitigation.
![alt text](https://user-images.githubusercontent.com/28737912/29939745-e8f58542-8e5a-11e7-9936-d5c1d7cde7c7.png) Figure 2. Prediciton error distribution for large earthquakes (> M4.5) using ridge regression, random forest and CNN compared to the baseline. 
