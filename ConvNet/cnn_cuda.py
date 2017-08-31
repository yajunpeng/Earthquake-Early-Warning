#!/usr/bin/env python3
from __future__ import print_function, division
import os
import sys
import time
import h5py
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from utils import dump_json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#plt.style.use('seaborn-darkgrid')
      
def load_data():
    """
    Load train and test data saved in disk.
    
    Returns
    -------   
    X: feature matrix (waveforms)
    y: magnitudes
    d: source-station distances
    """
    t1 = time.time()
    f = h5py.File("./Train_test_set_augment.h5")
    X_train = np.array(f["X_train"])
    y_train = np.array(f["y_train"])   
    X_test = np.array(f["X_test"])
    y_test = np.array(f["y_test"])
    d_train = np.array(f["train_distance"])
    d_test = np.array(f["test_distance"])
    X_train, X_test = feature_tranform(X_train, X_test,
                                       d_train, d_test,
                                       distance_correct = True)
    t2 = time.time()
    print("Time used in reading data: %.2f sec" % (t2 - t1))
    print("train x and y shape: ", X_train.shape, y_train.shape)
    print("test x and y shape: ", X_test.shape, y_test.shape)
    print("train d and test d shape: ", d_train.shape, d_test.shape)
    
    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
            "d_train": d_train, "d_test": d_test}  
    

def feature_tranform(_X_train, _X_test,
                     d_train, d_test,
                     distance_correct = True):
    """
    Rescale and correct for distance. 
    
    Parameters
    ----------
    _X_train: matrix like
        Feature matrix of the training set.     
    _X_test: matrix like
        Feature matrix of the test set.    
    _d_train: array like
        Source-statio distances (training).     
    _d_test: array like
        Source-statio distances (test).           
        
    Returns
    -------    
    Transformed X_train and X_test  
    """

    #Rescale features
    X_train = _X_train.copy()
    X_test = _X_test.copy()
    X_train[:,3:6,:] *= 20
    X_train[:,6:,:] *= 400
    X_test[:,3:6,:] *= 20
    X_test[:,6:,:] *= 400  
    
    if distance_correct:
        for i in range(X_train.shape[0]):
            X_train[i, :, :] *= (d_train[i] ** 2)
        for j in range(X_test.shape[0]):
            X_test[j, :, :] *= (d_test[j] ** 2)
    
    return X_train, X_test


def make_dataloader(xs, ys, cuda_flag=False):
    """
    Make pyTorch dataloader. 
    
    Parameters
    ----------
    xs: matrix like
        Feature matrix.
    ys: array like
        outcomes.
    cuda_flag:
        If true, use GPU.          
        
    Returns
    -------    
    loader: pyTorch dataloader.
    """
    
    if cuda_flag:
        xs = torch.Tensor(xs).cuda()
        ys = torch.Tensor(ys).cuda()
    else:
        xs = torch.Tensor(xs)
        ys = torch.Tensor(ys)
                    
    torch_dataset = Data.TensorDataset(
            data_tensor=xs, target_tensor=ys)
    loader = Data.DataLoader(
            dataset=torch_dataset, batch_size=batch_size, shuffle=True)
    
    return loader


def make_train_dev_test_loader(data_all, training_mode):
    """
    Make dataloader for the train, dev, test sets.
    
    Parameters
    ----------
    data_all: dict
        All data (Feature matrix and outcomes).
    training_mode: String
        If "Develop", 10% of training set are used
        as a dev set to tune hyperparameters.
        if "train", all training set data are used 
        to train a model.        
        
    Returns
    -------    
    train_loader, dev_loader, test_loader, 
    train_size (with or without a dev set)
    """
    
    if training_mode == "develope":    
        #develop mode: 10% training set is used as dev set.
        data_agument = 6
        num_eq = int(data_all["y_train"].shape[0] / data_agument)
        mag = data_all["y_train"].copy()[:num_eq]
        mag[mag < 4.5] = 0
        mag[mag >= 4.5] = 1
        X_train_index, X_dev_index, _, _ = \
            train_test_split(np.arange(num_eq), 
                             data_all["y_train"][:num_eq],
                             stratify=mag,
                             test_size=0.1, 
                             random_state=123)
        X_dev = data_all["X_train"][X_dev_index, :, :]
        y_dev = data_all["y_train"][X_dev_index]
        X_train = np.array([]).reshape(0, data_all["X_train"].shape[1], data_all["X_train"].shape[2])
        y_train = np.array([])
        for i in range(data_agument):
            index_train = X_train_index + i * num_eq
            X_train = np.vstack((X_train, data_all["X_train"][index_train, :, :]))
            y_train = np.append(y_train, data_all["y_train"][index_train])
        
        train_size = X_train.shape[0]
        print("Develop mode...")
        print("Train X and y shape: ", X_train.shape, y_train.shape)
        print("Dev X and y shape: ", X_dev.shape, y_dev.shape)
        
        train_loader = make_dataloader(X_train, 
                                       y_train,
                                       cuda_flag = cuda_flag)
        
        dev_loader = make_dataloader(X_dev, 
                                     y_dev,
                                     cuda_flag = cuda_flag)
        
        test_loader = None
        
    elif training_mode == "train":     
        #training mode: all training data are used    
        train_size = data_all["X_train"].shape[0]
        train_loader = make_dataloader(data_all["X_train"], 
                                       data_all["y_train"],
                                       cuda_flag = cuda_flag) 
        
        dev_loader = None
        
        test_loader = make_dataloader(data_all["X_test"], 
                                      data_all["y_test"],
                                      cuda_flag = cuda_flag)
    
    return train_loader, dev_loader, test_loader, train_size


class CNN(nn.Module):
    """
    Contruct a 10-layer ConvNet.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv1d(9, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU())
        self.layer2 = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU())
        self.layer3 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU())      
        self.layer5 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU())
        self.layer6 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2))      
        self.layer7 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU())
        self.layer8 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2))          
        self.fc1 = nn.Linear(10*32, 10*32)
        self.fc2 = nn.Linear(10*32, 1)
        self.num_layers = 10
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out        
    

def predict_y(net, data_loader):
    y_pred = []
    y_true = []
    
    for X_batch, y_batch in data_loader:
        if cuda_flag:
            X = Variable(X_batch).cuda()
            y = Variable(torch.Tensor([y_batch.numpy(), ])).cuda()
        else:
            X = Variable(X_batch)
            y = Variable(torch.Tensor([y_batch.numpy(), ]))
        prediction = net(X)
        if cuda_flag:
            y_pred.extend(
                    torch.squeeze(prediction.data).cpu().numpy().tolist()) 
            y_true.extend(
                    torch.squeeze(y.data).cpu().numpy().tolist())     
        else:
            y_pred.extend(
                    torch.squeeze(prediction.data).numpy().tolist()) 
            y_true.extend(
                    torch.squeeze(y.data).numpy().tolist())

    return y_pred, y_true

def calc_MSE(y_true, y_pred, mag_large = 4.5):
    """
    MSE for all earthquakes as well as for
    large earthquakes only. 
    """
    y_pred_large = \
        [y_pred[i] for i in range(len(y_true)) if y_true[i] >= mag_large]
    y_true_large = \
        [y_true[i] for i in range(len(y_true)) if y_true[i] >= mag_large]
    mse_all = mean_squared_error(y_true, y_pred)
    mse_large = mean_squared_error(y_true_large, y_pred_large)
    
    return mse_all, mse_large
    
    
def main(outputdir, model_num, 
         training_mode, train_size,
         train_loader, dev_loader, test_loader,
         learning_rate = 0.001, weight_decay = 0.0001):
    
    print("Training mode: ", training_mode)
    print("=== Model %d: learning_rate(%s) --- weight_decay(%s) ===" \
          % (model_num, learning_rate, weight_decay))
    print("Cuda flag: %s" % cuda_flag)
      
    net = CNN()
    if cuda_flag:
        net.cuda()
    print(net)
    
    optimizer = torch.optim.Adam(net.parameters(), 
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    loss_func = nn.MSELoss()
    
    # train
    nstep = train_size // batch_size + 1
    all_loss = {}
    for epoch in range(num_epochs):
        loss_epoch = []
        for step, (X_batch, y_batch) in enumerate(train_loader):
            if cuda_flag:
                X = Variable(X_batch).cuda()
                y = Variable(torch.Tensor([y_batch.numpy(), ])).cuda()
            else:
                X = Variable(X_batch)
                y = Variable(torch.Tensor([y_batch.numpy(), ]))
                
            optimizer.zero_grad()  # clear gradients for this training step
            prediction = net(X)
            loss = loss_func(prediction, y)            
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
            loss_epoch.append(loss.data[0])
            
            if ((step + 1) % 10) == 1:
                print('Epoch:  %d/%d' % (epoch, num_epochs), 
                      '| Step: %d/%d' % (step, nstep),
                      "| Loss: %.4f" % np.mean(loss_epoch))            
    
        all_loss["epoch_%d" % epoch] = loss_epoch
        print("=== Mean loss in epoch(%d): %f ==="
              % (epoch, np.mean(loss_epoch)))
    
    #prediction for train set
    y_pred_train, y_true_train = predict_y(net, train_loader)
    
    #prediction for test/dev set
    #(develope uses dev_loader, train uses test_loader)
    if training_mode == "develope":
        y_pred_test, y_true_test = predict_y(net, dev_loader)
    elif training_mode == "train":
        y_pred_test, y_true_test = predict_y(net, test_loader)        
    
    #prediction MSE (for all and those >M4.5)   
    _mse_all_train, _mse_large_train = \
        calc_MSE(y_true_train, y_pred_train, mag_large = 4.5)
    _mse_all_test, _mse_large_test = \
        calc_MSE(y_true_test, y_pred_test, mag_large = 4.5)     
    
    #save prediction
    filename = training_mode + "_model_"+ str(model_num) + "_prediction.json"
    outputfn = os.path.join(outputdir, filename)
    print("output file: %s" % outputfn)
    output = {"learning_rate": learning_rate,
              "weight_decay": weight_decay,
              "mse_all_train": _mse_all_train,
              "mse_all_dev_test": _mse_all_test,
              "mse_large_train": _mse_large_train,
              "mse_large_test": _mse_large_test,
              "epoch_loss": all_loss, 
              "y_pred_train": y_pred_train, 
              "y_true_train": y_true_train,
              "y_pred_test": y_pred_test, 
              "y_true_test": y_true_test}
    dump_json(output, outputfn)




if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("input training mode (train or develope)...exit")
        sys.exit()

    training_mode = sys.argv[1]
    
    outputdir = './output/'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        
    num_epochs = 20
    torch.manual_seed(123)  # reproducible
    cuda_flag = torch.cuda.is_available()
    batch_size = 128    
        
    data_all = load_data()   
    train_loader, dev_loader, test_loader, train_size = \
        make_train_dev_test_loader(data_all, training_mode) 
    
    # Train multiple models with random learning rate 
    # and weight decay.
    if training_mode == "develope":
        learning_rate_list = 10**(np.random.rand(80) * 4 - 5)
        weight_decay_list = 10**(np.random.rand(80) * 4 - 5)  
        
    elif training_mode == "train":
        learning_rate_list = [0.0007]
        weight_decay_list = [0.00006]
        
    else:
        raise ValueError("Error in training mode name: %s," % training_mode, 
                         "Choose from: train (training) or develope (tuning)")  
        
          
    for model_num in range(len(learning_rate_list)):
        
        learning_rate = learning_rate_list[model_num]
        weight_decay = weight_decay_list[model_num]
        main(outputdir, model_num,
             training_mode, train_size,
             train_loader, dev_loader, test_loader,
             learning_rate, weight_decay)
    
    
    
    
    
    
