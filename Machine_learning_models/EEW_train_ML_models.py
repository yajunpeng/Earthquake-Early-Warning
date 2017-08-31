#!/usr/bin/env python3
from __future__ import print_function, division
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time
import pandas as pd
from sklearn.metrics import mean_squared_error
from regressor import train_ridge_linear_model, train_lasso_model, train_EN_model
from regressor import train_random_forest_model
from regressor import train_adaboosted_RF_model
from regressor import dimension_reduction_with_PCA, dimension_reduction_with_MI
from utils import dump_json
from utils import standarize_feature

plt.style.use('seaborn-darkgrid')
                          
def split_train_and_test_set(measurements, 
                             min_mag = 2.8, 
                             mag_large = 4.5,
                             test_size = 0.25, 
                             save_flag = True, 
                             save_dir = './'):
    """
    Measurements are divided into a train set and a test set.
    Data of the same earthquake at different stations are 
    included in the same set.
    
    Parameters
    ----------
    measurements: pandas dataframe
        Original features and outcomes.
    min_mag: float
        Minimum magnitude allowed.
    mag_large: float
        Earthquakes above mag_stratify are considered
        large earthquakes. 
    test_size: float
        The proportion for test set.
    save_flag: boolean
        Save the splitted data.
    save_dir: string
        Directory for the saved data.
        
    Returns
    -------    
    Train set and test set: pandas dataframe   
    """
    #clean data  
    measurements.drop(['Unnamed: 0'], axis = 1, inplace = True)                        
    measurements = measurements.loc[measurements['Z.disp.max_amp'] < 1e-2]
    measurements = measurements.loc[measurements['magnitude'] >= min_mag]
    
    #Split data to train set and test set 
    sources = measurements.source.unique()
    mag = np.zeros(len(sources))
    for i_source in range(len(sources)):
        mag[i_source] = measurements.loc[
                measurements['source'] == sources[i_source]].magnitude.iloc[0]
    mag[mag < mag_large] = 0
    mag[mag >= mag_large] = 1
    
    sources_train, sources_test, mag_train, mag_test = \
    train_test_split(sources, mag, test_size=test_size, 
                     stratify=mag, random_state=678)
    
    measurements['train_test'] = 1
    for source in sources_test:
        measurements.loc[measurements['source'] == source, 'train_test']= 0
    
    train_set = measurements.copy().loc[measurements['train_test'] == 1]
    test_set = measurements.copy().loc[measurements['train_test'] == 0]
    
    train_set.dropna(axis = 0, how = 'any', inplace = True)
    test_set.dropna(axis = 0, how = 'any', inplace = True)
    train_set.drop(['train_test'], axis = 1, inplace = True) 
    test_set.drop(['train_test'], axis = 1, inplace = True) 
    measurements.drop(['train_test'], axis = 1, inplace = True)     
    
    print("Split the measurements of size {0} to train set {1} and test set {2}".format(
            measurements.shape, train_set.shape, test_set.shape))
    print("%.2f%% of the train set are large earthquakes (>= M4.5)" % 
          ((np.sum(train_set.magnitude > mag_large)/len(train_set))*100))
    print("%.2f%% of the test set are large earthquakes (>= M4.5)" % 
      ((np.sum(test_set.magnitude > mag_large)/len(test_set))*100))
    
    if save_flag:
        train_set.to_hdf(save_dir + 'train_set.hdf5', 'df')
        test_set.to_hdf(save_dir + 'test_set.hdf5', 'df')    
    
    return train_set, test_set
        
  
def feature_engineering(feature, distance_correction = True):    
    """
    Remain linear: 'kurt', 'skew', 'max_amp_loc'    
    Logarithm: 'tau_c', 'tau_p_max', 'freq'    
    Logarithm and correction for geometrical 
    decay of amplitudes (* distance^2): the rest   
    
    Parameters
    ----------
    feature: pandas dataframe
        The feature matrix.
    distance_correction: boolean
        Correct for the earthquake-station distance.        
        
    Returns
    -------    
    The feature matix after engineering    
    """
    
    feature_linear = []
    feature_log = []
    
    feature_linear_string = ['kurt', 'skew', 'max_amp_loc']
    for string in feature_linear_string:
        feature_linear.extend(list(feature.filter(like=string).columns))

    feature_log_string = ['tau_c', 'tau_p_max', 'freq']
    for string in feature_log_string:
        feature_log.extend(list(feature.filter(like=string).columns)) 
    
    distance = feature.copy()['distance']  
    for col in feature.columns:
        if (col == 'distance') | (col in feature_linear):
            continue
        elif col in feature_log:
            feature[col] = np.log10(feature[col])
        else:
            if distance_correction:
                feature[col] = np.log10(feature[col]) + 2 * np.log10(distance) 
            else:
                feature[col] = np.log10(feature[col])              
    feature['distance'] = np.log10(feature['distance'])  
    
    if not distance_correction:
        feature.drop(['distance'], axis = 1, inplace = True) 
         
    return feature    
  

def make_feature_matrix_and_outcome(data_set):
    """
    Make X (features) and y (magnitude).

    Parameters
    ----------
    data_set: pandas dataframe
        The train/test set.    
        
    Returns
    -------    
    X and y: pandas dataframe     
    """
    outcome = data_set.copy()['magnitude']
    feature = data_set.copy().drop(['source', 'magnitude', 'channel'], 
                            axis = 1)
    feature = feature_engineering(feature)
    
    return feature, outcome


def plot_y(y_train, y_train_pred, y_test, y_test_pred, figname=None):
    """
    A quick plot for true y vs. predicted y.
    """
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.2, label="train")
    plt.plot([2, 8], [2, 8], '--', color="k")
    plt.title("Train")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, color="r", alpha=0.2, label="test")
    plt.plot([2, 8], [2, 8], '--', color="k")
    plt.title("Test")
    plt.legend()
    
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)   


def load_data():
    """
    Load measurement data and make train and test set.
    """
    
    data_path = '../data/'     
    start = time.time()         
    try:                                    
        print('Loading preprocessed measurement data saved in disk...')    
        train_set = pd.read_hdf('./train_set.hdf5')
        test_set = pd.read_hdf('./test_set.hdf5')
    
    except IOError:    
        print('No saved data.')        
        print('Loading and preprocessing raw measurement data...')
        measurements = pd.read_csv(data_path + "measurements.csv")                                                                                     
        train_set, test_set = split_train_and_test_set(measurements)                                       
    
    X_train, y_train = make_feature_matrix_and_outcome(train_set)
    X_test, y_test = make_feature_matrix_and_outcome(test_set)

    end = time.time()
    print('Data loading and feature engineering: completed. Time: %d s' % (end - start))
    
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test,
            "y_test": y_test}
    


def train_ML_model(_X_train, y_train, 
                   _X_test, y_test,
                   X_name, model = "ridge", 
                   PCA_flag = False, 
                   MI_flag = False, 
                   n_components = 300,
                   plot_flag = False,
                   outputdir = "./output/"):
    """
    Train machine learning models.
    Choose from: 
        Linear regression: Ridge, Lasso, EN (elastic net)
        SVM
        RF (Random Forest)
        Ada_RF_exp (adaboosted random forest, exponential loss)
    
    Parameters
    ----------
    _X_train and _X_test: pandas dataframe
        Train and test feature matrix  
    y_train and y_test: pandas dataframe
        Train and test outcome (magnitude)
    X_name: list
        Feature names
    model: string
        Selected machine learning model
    PCA_flag and MI_flag: boolean
        Use PCA or mutual information 
        for feature selection
    n_components: int
        Choose first n_components features
        based on PCA or MI 
    plot_flag: boolean
        A quick plot for true y and 
        predicted y
    outputdir: string
        Output directory        
    """    
    
    # train and test set are standardized.
    X_train, X_test = \
        standarize_feature(_X_train, _X_test)    
    print("train X and y shape: ", X_train.shape, y_train.shape)
    print("test X and y shape: ", X_test.shape, y_test.shape)  
    
    # Dimension reduction for the feature matrix using Principal 
    # component analysis and mutual information.
    # Not very useful here. 
    if PCA_flag:
        X_train, X_test = \
            dimension_reduction_with_PCA(X_train, X_test, n_components)  
        print("After dimension reduction, train X and y shape: ", 
              X_train.shape, y_train.shape)
        print("After dimension reduction, test X and y shape: ", 
              X_test.shape, y_test.shape)            
                
    elif MI_flag:
        X_train, X_test = \
            dimension_reduction_with_MI(_X_train, _X_test, y_train, 
                                        X_name, n_components)
        print("After dimension reduction, train X and y shape: ", 
              X_train.shape, y_train.shape)
        print("After dimension reduction, test X and y shape: ", 
              X_test.shape, y_test.shape)            
    
    # Train machine learning models    
    if model.lower() == "ridge":
        # Ridge regression
        info = train_ridge_linear_model(
            X_train, y_train, X_test)
        
    elif model.lower() == "lasso":
        # Lasso regression
        info = train_lasso_model(X_train, y_train, X_test)
        
    elif model.lower() == "en":
        # Elastic net regression (a mix of ridge and lasso)
        info = train_EN_model(X_train, y_train, X_test)
        
    elif model.lower() == "rf":
        # Random Forest
        info = train_random_forest_model(X_train, y_train, X_test) 
        
    elif model.lower() == 'ada_rf_exp':
        # Adaboosted random forest (exponential loss)
        info = train_adaboosted_RF_model(X_train, y_train, X_test) 
        
    else:
        raise ValueError("Error in model name: %s," % model, 
                         "Choose from: ridge, lasso, en, rf, ada_rf")

    print("y_test and y_test_pred: ", y_test.shape, info["y"].shape)
    _mse = mean_squared_error(y_test, info["y"])
    _std = np.std(y_test - info["y"])
    print("MSE on test data: %f" % _mse)
    print("std of error on test data: %f" % _std)

    if PCA_flag:
        file_prefix = "%s_PCA_%d_components" % (model, n_components)
    elif MI_flag:
        file_prefix = "%s_MI_%d_components" % (model, n_components)            
    else:
        file_prefix = "%s" % (model)
        
    figname = os.path.join(outputdir, "%s.png" % file_prefix)
    
    if plot_flag:
        print("save figure to file: %s" % figname)
        plot_y(y_train, info["y_train"], y_test, info["y"], figname=figname)

    # Output:
    #    feature names; feature coefficient or importance;
    #    true y and predicted y (train and test); 
    #    test MSE; test error std
    content = {"features": X_name,
               "coef": list(info["coef"]),
               "y_train": list(y_train),
               "y_train_pred": list(info["y_train"]),
               "y_test": list(y_test),
               "y_test_pred": list(info["y"]),
               "MSE": _mse, "error_std": _std}
    
    outputfn = os.path.join(outputdir, "%s.json" % file_prefix)
    print("save results to file: %s" % outputfn)
    dump_json(content, outputfn)
    
    
def main(model, outputdir):
    """
    Load data and train machine learning models.
    """
    print("model: %s" % (model))
    print("outputdir: %s" % outputdir)
    
    data = load_data()
    X_train = data['X_train'].copy().values 
    y_train = data["y_train"].copy().values 
    X_test = data['X_test'].copy().values
    y_test = data["y_test"].copy().values
    X_name = list(data['X_train'].columns)
        
    train_ML_model(X_train, y_train, X_test, y_test, X_name,
                  model = model, PCA_flag = False,
                  MI_flag = False, outputdir = outputdir)
    
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Missing arguement: model")

    model = sys.argv[1]
    outputdir = "./output/"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    main(model, outputdir)
