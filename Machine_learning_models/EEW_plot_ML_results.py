#!/usr/bin/env python3
from utils import load_json
from utils import plot_stations_and_events, plot_map_us
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from EEW_train_ML_models import load_data
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from obspy import read
plt.style.use('seaborn-darkgrid')


def error_large_eq(y_true, y_pred, large_mag = 4.5):
    y_true_large = np.asarray([y_true[i] for i in 
                               range(len(y_true)) if y_true[i] >= large_mag])
    y_pred_large = np.asarray([y_pred[i] for i in 
                               range(len(y_true)) if y_true[i] >= large_mag])
    y_true_small = np.asarray([y_true[i] for i in 
                               range(len(y_true)) if y_true[i] < large_mag])
    y_pred_small = np.asarray([y_pred[i] for i in 
                               range(len(y_true)) if y_true[i] < large_mag])
    
    MSE = mean_squared_error(y_true, y_pred) 
    MSE_large = mean_squared_error(y_true_large, y_pred_large)    
    MSE_small = mean_squared_error(y_true_small, y_pred_small)  
    mean_error_large = np.mean(y_pred_large-y_true_large) 
    mean_error_small = np.mean(y_pred_small-y_true_small) 
    std_error_large = np.std(y_pred_large-y_true_large) 
    std_error_small = np.std(y_pred_small-y_true_small)     
    
    return {"MSE": MSE,
            "MSE_large": MSE_large, 
            "MSE_small": MSE_small, 
            "mean_error_large": mean_error_large, 
            "mean_error_small": mean_error_small,
            "std_error_large": std_error_large, 
            "std_error_small": std_error_small,
            "error_large": y_pred_large - y_true_large,
            "error_small": y_pred_small - y_true_small}  
    
def plot_true_vs_prediction(y_true, y_pred,
                            symbol = '.'):
    plt.plot(y_true, y_pred, symbol, alpha = 0.5)
    plt.plot([2, 6.5], [2, 6.5], 'k--', linewidth = 2)
    plt.xlim([2, 6.5])
    plt.ylim([2, 6.5])
    
def plot_true_vs_error(y_true, y_pred, error,
                       symbol = 'k.', label_font = 25):
    plt.plot(y_true, y_pred - y_true, symbol, alpha = 0.5)
    plt.text(2.2, -1.7, 'MSE = %.4f' %(error['MSE']), fontsize = label_font)
    plt.text(2.2, -2, 'MSE (> M4.5) = %.4f' %(error['MSE_large']), 
             fontsize = label_font)
    plt.text(2.2, -2.3, r'Error (> M4.5) = %.4f $\pm$ %.4f' %(
            error['mean_error_large'], error['std_error_large']), 
            fontsize = label_font)
    plt.plot([2, 6.5], [0, 0], 'k--', linewidth = 2)
    plt.ylim([-2.5, 1.5])
    plt.xlim([2, 6.5])    

def plot_feature_importance(coef, feature_name, ax, top_feature = 50):
    coef_index_top = np.argsort(np.abs(coef))[::-1]
    feature_top = [feature_name[i]for i in coef_index_top[:top_feature]]
    feature_coef_top = [coef[i] for i in coef_index_top[:top_feature]]
    y_pos = np.arange(top_feature)[::-1]
    plt.barh(y_pos, feature_coef_top, align='center', alpha=0.4)
    ax.yaxis.tick_right()
    plt.yticks(y_pos, feature_top,fontsize=10)
    plt.ylim(-1,top_feature+0.25)    


def main():
    outputdir = './output/'
    data = load_data()
    X_train = data['X_train']
    y_train = data["y_train"]
    X_test = data['X_test']
    y_test = data["y_test"]
      
    ridge = load_json(outputdir + 'ridge.json')
    ada_rf_exp = load_json(outputdir + 'ada_rf_exp.json')
    #rf = load_json(outputdir + 'rf.json')
    
    # Baseline model (linear fit for log10(Z.disp.max_amp))
    z = np.polyfit(X_train['Z.disp.max_amp'].values, y_train.values, 1)
    y_pred_base = X_test['Z.disp.max_amp'].values * z[0] + z[1]
    
    
    #####################################################
    #Figure 1:
    #Empirical measurements used in literature
    #####################################################
    label_font = 20
    tick_font = 20
    alpha = 0.5
    plt.figure(1, figsize = (24, 5))
    fig = plt.subplot(131)
    plt.plot(y_train, X_train['Z.disp.max_amp'], '.', alpha=alpha)
    plt.plot(y_test, X_test['Z.disp.max_amp'], 'r.', alpha=alpha)
    plt.plot(np.arange(2.5, 7.5, 1), (np.arange(2.5, 7.5, 1)-z[1])/z[0], 'k--', linewidth = 2)
    plt.xlabel('Magnitude', fontsize = label_font)
    plt.ylabel(r'$log_{10}(P_d * r^2)$', fontsize = label_font)
    fig.tick_params(labelsize = tick_font)
    fig.legend(['Train', 'Test', 'Linear fit (train)'], loc = 'lower right', fontsize = 15)
    
    fig = plt.subplot(132)
    plt.plot(y_train, X_train['Z.tau_c'], '.', alpha=alpha)
    plt.plot(y_test, X_test['Z.tau_c'], 'r.', alpha=alpha)
    plt.xlabel('Magnitude', fontsize = label_font)
    plt.ylabel(r'$log_{10}(\tau_c)$', fontsize = label_font)
    fig.tick_params(labelsize = tick_font)
    
    fig = plt.subplot(133)
    plt.plot(y_train, X_train['Z.tau_p_max'], '.', alpha=alpha)
    plt.plot(y_test, X_test['Z.tau_p_max'], 'r.', alpha=alpha)
    plt.xlabel('Magnitude', fontsize = label_font)
    plt.ylabel(r'$log_{10}(\tau_p^{max})$', fontsize = label_font)
    #plt.ylabel(r'$log_{10}(mean frequency)$', fontsize = label_font)
    fig.tick_params(labelsize = tick_font)
    
    plt.savefig(outputdir + 'empirical_measurements.png', format = 'png')
    
    
    #####################################################
    #Figure 2:
    #Feature coefficients/importance 
    #####################################################
    plt.figure(2, figsize=(12,8))
    
    ax=plt.subplot(141) 
    plot_feature_importance(ridge['coef'], ridge['features'], ax, top_feature = 30)
    plt.xlabel('Coefficient', fontsize = 15)
    plt.title('Ridge', fontsize = 15)
    #plt.tick_params(labelsize = tick_font)
    ax=plt.subplot(143) 
    plot_feature_importance(ada_rf_exp['coef'], ada_rf_exp['features'], ax, top_feature = 30)
    plt.xlabel('Feature importance', fontsize = 15)
    plt.title('Adaboosted Random Forest', fontsize = 15)
    #plt.tick_params(labelsize = tick_font)
    
    plt.savefig(outputdir + 'Feature_importance.png', format='png')
    
    
    #####################################################
    #Figure 3:
    #True magnitude vs. prediction 
    #####################################################
    label_font = 25
    tick_font = 20
    alpha = 0.5
    
    plt.figure(3, figsize = (27, 15))
    
    fig = plt.subplot(2,3,1)
    plot_true_vs_prediction(y_test, y_pred_base,
                            symbol = 'r.')
    plt.text(2.2, 6.0, 'Test set', fontsize = label_font + 10)
    fig.tick_params(labelsize = tick_font + 5)
    plt.ylabel('Predicted magnitude', fontsize = label_font + 10)  
    plt.title('Baseline', fontsize = label_font + 10)
    
    fig = plt.subplot(2,3,2)
    plot_true_vs_prediction(y_test, ridge['y_test_pred'],
                            symbol = 'r.')
    fig.tick_params(labelsize = tick_font + 5)
    plt.title('Ridge', fontsize = label_font + 10)
    
    fig = plt.subplot(2,3,3)
    plot_true_vs_prediction(y_test, ada_rf_exp['y_test_pred'],
                            symbol = 'r.')
    fig.tick_params(labelsize = tick_font + 5)
    plt.title('Adaboosted Random Forest', fontsize = label_font + 10)
    
    #####################################################
    
    fig = plt.subplot(2,3,4)
    error_base = error_large_eq(y_test.values, y_pred_base)
    plot_true_vs_error(y_test.values, y_pred_base, error_base)
    plt.xlabel('True Magnitude', fontsize = label_font + 10)
    plt.ylabel('Test error', fontsize = label_font + 10) 
    fig.tick_params(labelsize = tick_font + 5)
    
    fig = plt.subplot(2,3,5)
    error_ridge = error_large_eq(y_test.values, ridge['y_test_pred'])
    plot_true_vs_error(y_test.values, ridge['y_test_pred'], error_ridge)
    plt.xlabel('True Magnitude', fontsize = label_font + 10)
    fig.tick_params(labelsize = tick_font + 5)
    
    fig = plt.subplot(2,3,6)
    error_ada_rf = error_large_eq(y_test.values, ada_rf_exp['y_test_pred'])
    plot_true_vs_error(y_test.values, ada_rf_exp['y_test_pred'], error_ada_rf)
    plt.xlabel('True Magnitude', fontsize = label_font + 10)
    fig.tick_params(labelsize = tick_font + 5)
    
    plt.savefig(outputdir + 'prediction.png', format='png')
    
    
    #####################################################
    #Figure 4:
    #Error distribution (> M4.5)
    #####################################################
    plt.figure(4, figsize = (8, 5.5))
    bins = np.arange(-1.4, 0.8, 0.1)
    plt.hist(error_base['error_large'], bins=bins, alpha=0.7, label="Baseline")
    plt.hist(error_ridge['error_large'],bins=bins, alpha=0.7, label="Ridge")
    plt.hist(error_ada_rf['error_large'],bins=bins, alpha=0.7, label="Adaboosted RF")
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.plot([0, 0],[0, 50],'k--')
    plt.ylim([0, 50])
    plt.xlabel('Error (> M4.5)', fontsize = label_font)
    plt.tick_params(labelsize = tick_font)
    plt.quiver(-0.05, 42, -1, 0)
    plt.quiver(0.05, 42, 1, 0)
    plt.text(-0.6, 45, 'Underestimate', fontsize=15)
    plt.text(0.12, 45, 'Overestimate', fontsize=15)    
    
    plt.savefig(outputdir + 'error.png', format='png')
    
    
    #####################################################
    #Figure 5:
    #Map view
    #####################################################    
    
    station_loc = pd.read_csv('../data/station.csv')
    catalog = pd.read_csv('../data/source.csv')
    plot_stations_and_events(station_loc, catalog, map_flag = True)
    plt.savefig(outputdir + 'mapview.png', format='png')
    
    plot_map_us()
    plt.savefig(outputdir + 'mapview_us.png', format='png')
    
    #####################################################
    #Figure 6:
    #Time series data visualization
    #####################################################
    
    st = read('../data/proc/2016-06-10T08:04:38.700000Z/CI.mseed')
    trace = st.select(channel='BHE', station='BAR')
    df = trace[0].stats.sampling_rate
    cft = recursive_sta_lta(
        trace[0], int(0.05 * df), int(20 * df))
    arrivals = trigger_onset(cft, 100, 0.5)
    start = arrivals[0][0]/df-60
    
    plt.figure(5, figsize = (12, 6))
    #plt.subplot(2,1,1)
    plt.fill_between([start, start+4], -0.0075, 0.009, 
                     facecolor = [0.7, 0.7, 0.7])
    plt.plot(np.arange(0, 180, 1./df)-60, trace[0])
    plt.xlim([60-60, 130-60])
    plt.ylim([-0.0075, 0.009])
    plt.text(start+0.25, -0.006, '4 s', fontsize=25)
    plt.tick_params(labelsize = tick_font)
    plt.title('M5.19, 2016-06-10, 08:04:38.70', fontsize = label_font)
    plt.xlabel('Time (s)', fontsize=label_font)
    plt.ylabel('Amplitude (m/s)', fontsize=label_font)
    plt.savefig(outputdir + 'time_series.png', format='png')
    #
    #plt.subplot(2,1,2)
    #plt.fill_between([start, start+4], -5, 320, 
    #                 facecolor = [0.7, 0.7, 0.7])
    #plt.plot(np.arange(0, 180, 1./df), cft)
    #plt.plot([60, 130], [100, 100], 'k--', linewidth=2)
    #plt.xlim([60, 130])
    #plt.tick_params(labelsize = tick_font)



if __name__ == "__main__":
    main()
    













