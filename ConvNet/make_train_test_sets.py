#!/usr/bin/env python3
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

def split_train_and_test_stratify(source_info, 
                                  magnitude,
                                  waveform,
                                  distance,
                                  mag_stratify = 4.5,
                                  test_size = 0.25,
                                  augment_flag = True,
                                  save_flag = True, 
                                  save_dir = './'):
    """
    Split the dataset to a training and a test set.
    
    Parameters
    ----------
    source_info: 
        Earthquake information (e.g. time, location,
        magnitude)
    magnitude: float
        Earthquake magnitudes
    waveform: float, array like
        Seismic waveforms.
    mag_stratify: float
        Earthquakes with magnitude >= mag_stratify
        are large earthquakes. Used for stratifying
        when splitting train and test sets.
    test_size: float
        The proportion for test set.
    augment_flag: boolean
        If true, augment the data size (6X) by time 
        shifting and adding random noise.
    save_flag: boolean
        Save the splitted data.
    save_dir: string
        Directory for the saved data.
        
    Returns
    -------    
    Train set and test set: array like 
    """

    _source_info = source_info.copy().iloc[::-1]
    sources = _source_info.source.unique()
    mag = np.zeros(len(sources))
    for i_source in range(len(sources)):
         eq_index = _source_info.loc[
                 _source_info['source'] == sources[i_source]].index[0]
         mag[i_source] = magnitude[eq_index]
         
    mag[mag < mag_stratify] = 0
    mag[mag >= mag_stratify] = 1
    
    sources_train, sources_test, _, _ = \
    train_test_split(sources, mag, test_size=test_size, 
                     stratify=mag, random_state=678)
    
    source_info['train_test'] = True
    for source in sources_test:
        source_info.loc[source_info['source'] == source, 'train_test']= False
        
    msk = np.asarray(source_info['train_test'])
    _X_train = waveform.copy()[msk]
    _X_test = waveform.copy()[~msk]
    _y_train = magnitude.copy()[msk]
    y_test = magnitude.copy()[~msk]  
    _d_train = distance.copy()[msk]
    d_test = distance.copy()[~msk]
    
    X_test = _X_test.copy()[:, :, 20:100]
    if not augment_flag:
        print('No data augmentation.')
        X_train = _X_train.copy()[:, :, 20:100]        
        y_train = _y_train.copy()
        d_train = _d_train.copy()
    else:
        print('Augmenting training data (time shift and Gaussian noise).')
        X_train = np.array([]).reshape(0, _X_train.shape[1], _X_train.shape[2]-40)
        y_train = np.array([])
        d_train = np.array([])
        print("X_train, y_train, d_train shape:", 
              X_train.shape, y_train.shape, d_train.shape)         
        for shift in range(-5,10,5):
            _X_train_temp = _X_train.copy()[:, :, 20+shift:100+shift]
            _X_train_temp_noise = _X_train_temp.copy()
            for d1 in range(_X_train_temp_noise.shape[0]):
                for d2 in range(_X_train_temp_noise.shape[1]):
                    _X_train_temp_noise[d1, d2, :] += \
                            0.1 * np.std(_X_train_temp_noise[d1, d2, :]) * \
                            np.random.randn(_X_train_temp_noise.shape[2])
                           
            X_train = np.vstack((X_train, _X_train_temp, _X_train_temp_noise))
            y_train = np.append(y_train, _y_train)
            y_train = np.append(y_train, _y_train)
            d_train = np.append(d_train, _d_train)
            d_train = np.append(d_train, _d_train)
            print("X_train, y_train, d_train shape:", 
                  X_train.shape, y_train.shape, d_train.shape) 
        
    print("train X and y shape: ", X_train.shape, y_train.shape)
    print("test X and y shape: ", X_test.shape, y_test.shape)
    print("train dist and test dist shape: ", d_train.shape, d_test.shape)  
    
    if save_flag:
        if augment_flag:
            outputfn = save_dir + 'Train_test_set_augment.h5'
        else:
            outputfn = save_dir + 'Train_test_set.h5'                       
        
        f = h5py.File(outputfn, 'w')
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)
        f.create_dataset("train_distance", data=d_train)
        f.create_dataset("test_distance", data=d_test)
        f.close()
        print('Data saved: ', outputfn)
    else:
        return X_train, X_test, y_train, y_test, d_train, d_test

 
def main():    
    data_path = '../data/'
    f = h5py.File(data_path + "dataset_nn.h5")
    waveform = np.array(f["waveform"])
    magnitude = np.array(f["magnitude"])
    distance = np.array(f["distance"])
    f.close()
    source_info = pd.read_csv(data_path + "dataset_source_nn.csv")     
    print("input waveform, distance, and magnitude shape: ", waveform.shape,
          distance.shape, magnitude.shape)
    
    split_train_and_test_stratify(source_info, 
                                  magnitude,
                                  waveform,
                                  distance,
                                  mag_stratify = 4.5,
                                  test_size = 0.25,
                                  augment_flag = True,
                                  save_flag = True, 
                                  save_dir = './')


if __name__ == "__main__":
    main()