"""
1) Load data windows for each station and each event
2) Use 4s-long seismograms to generate measurements
"""
from __future__ import print_function, division
import os
import sys
import scipy.stats
import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from utils import load_json
from metric import cut_window, tau_c, tau_p_max, \
    make_disp_vel_acc_records
from metric import norm, envelope, split_accumul_data

plt.style.use('seaborn-darkgrid')


def read_waveform(fn):
    """
    Read seismic data using obspy.
    
    Parameters
    ----------
    fn: string
        File name.
        
    Returns
    -------    
    st: obspy stream
        Seismic data.
    """   
    
    st = obspy.read(fn, format="MSEED")
    return st


def measure_func(data, sampling_rate, prefix, env = False):
    """
    Feature extraction. Designed features for each trace: 
        
        Descriptive statistics:
        L1, L2, L4 norm; maximum amplitude and its arrival time;
        mean; median; variance; skewness; kurtosis; 25, 50, 75, 90 percentile;
        
        Frequency-related features (Hammer et al., BSSA, 2012):
        Mean frequency; frequency bandwidth (std);
        Central frequency.   
    
    Parameters
    ----------
    data: array-like
        Seismic data.
    sampling_rate: int, float
        Instrument sampling rate.
    prefix: string
        Prefix for feature name.
    env: boolean
        Using waveform envelope or not.
        
    Returns
    -------    
    measure: dict
        Features on a trace.             
    """
    
    measure = {}
    npts = len(data)
    measure["%s.l1_norm" % prefix] = norm(data, ord=1) / npts
    measure["%s.abs_l1_norm" % prefix] = norm(np.abs(data), ord=1) / npts
    measure["%s.l2_norm" % prefix] = norm(data, ord=2) / npts
    measure["%s.l4_norm" % prefix] = norm(data, ord=4) / npts
    max_amp = np.max(np.abs(data))
    measure["%s.max_amp" % prefix] = max_amp
    max_amp_loc = np.argmax(np.abs(data)) / npts
    measure["%s.max_amp_loc" % prefix] = max_amp_loc
    #measure["%s.max_amp_over_loc" % prefix] = max_amp / max_amp_loc

    _, _, mean, var, skew, kurt = scipy.stats.describe(np.abs(data))
    measure["%s.mean" % prefix] = mean
    measure["%s.var" % prefix] = var
    measure["%s.skew" % prefix] = skew
    measure["%s.kurt" % prefix] = kurt
    for perc in [25, 50, 75, 90]:
        measure["%s.%d_perc" % (prefix, perc)] = \
            np.percentile(np.abs(data), perc)
    
    if not env:
        sp_abs = np.abs(np.fft.fft(data))[:int(npts/2)]
        sp_abs2 = (sp_abs ** 2)
        freq = (np.fft.fftfreq(npts, d = 1./ sampling_rate))[:int(npts/2)]
        #measure["%s.predominant_freq" % prefix] = freq[np.argmax(sp_abs2)]
        freq_mean = np.sum(freq * sp_abs2) / np.sum(sp_abs2)
        #print(freq_mean)
        measure["%s.mean_freq" % prefix] = freq_mean
        measure["%s.bandwidth_freq" % prefix] = np.sqrt(
                np.sum(((freq - freq_mean) ** 2) * sp_abs2) / np.sum(sp_abs2))
        measure["%s.central_freq" % prefix] = np.sqrt(
                np.sum((freq ** 2) * sp_abs2) / np.sum(sp_abs2))
    
    return measure


def measure_on_trace_data_type(trace, data_type, window_split):
    """
    Features made from original trace (disp, vel or acc) and its envelope.
    Use 8 windows (0.5s to 4s with increments of 0.5s, since the picked arrival time). 
    
    Parameters
    ----------
    trace: obspy trace
        Seismic data.
    data_type: string
        displacement, velocity or acceleration
    window_split: int
        Split the data into increasingly long windows
        
    Returns
    -------    
    measure: dict
        Features on a trace.   
    """
    
    measure = {}

    channel = trace.stats.channel[2]
    sampling_rate = trace.stats.sampling_rate
    prefix = "%s.%s" % (channel, data_type)

    measure.update(measure_func(trace.data, sampling_rate, prefix))

    data_split = split_accumul_data(trace.data, n=window_split)
    for idx in range(len(data_split)):
        prefix = "%s.%s.acumul_window_%d" % (channel, data_type, idx)
        measure.update(measure_func(data_split[idx], sampling_rate, prefix))

    env_data = envelope(trace.data)
    prefix = "%s.%s.env" % (channel, data_type)
    measure.update(measure_func(env_data, sampling_rate, prefix, env = True))

    env_split = split_accumul_data(env_data, n=window_split)
    for idx in range(len(data_split)):
        prefix = "%s.%s.env.acumul_window_%d" % (channel, data_type, idx)
        measure.update(measure_func(env_split[idx], sampling_rate, prefix, env = True))

    return measure


def plot_arrival_window(trace, windows, origin_time):
    """
    Plot trace data.
    
    Parameters
    ----------
    trace: obspy trace
        Seismic data.
    windows: dict
        picked arrivals
    origin_time: UTCDateTime
        Earthquake occurrence time.
 
    """    
    
    plt.plot(trace.data)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()

    idx = (UTCDateTime(windows["pick_arrival"]) - trace.stats.starttime) / \
        trace.stats.delta
    plt.vlines([idx], ymin, ymax, linestyles="dotted", color='r')

    idx = (UTCDateTime(windows["theo_arrival"]) - trace.stats.starttime) / \
        trace.stats.delta
    plt.vlines([idx], ymin, ymax, linestyles="dotted", color='b')

    idx = (origin_time - trace.stats.starttime) / \
        trace.stats.delta
    plt.vlines([idx], ymin, ymax, linestyles="dotted", color='b')

    plt.show()


def measure_tau_c(trace_types, window_split):
    """
    Tau_c is an emprical estimate for predominant frequency
    widely used in EEW literature.
    
    Parameters
    ----------
    trace_types: obspy trace
        Disp, vel, acc data
    window_split: int
        Split the data into increasingly long windows
        
    Returns
    -------    
    measure: dict
        Tau_c for a trace. 
    """
    channel = trace_types["disp"].stats.channel[2]

    measure = {}
    measure["%s.tau_c" % channel] = \
        tau_c(trace_types["disp"].data, trace_types["vel"].data)

    disp_split = split_accumul_data(trace_types["disp"].data, n=window_split)
    vel_split = split_accumul_data(trace_types["vel"].data, n=window_split)
    for idx in range(len(disp_split)):
        measure["%s.tau_c.accumul_window_%d" % (channel, idx)] = \
            tau_c(disp_split[idx], vel_split[idx])

    return measure


def measure_on_trace(_trace, windows, src, win_len=4.0, window_split=8):
    """
    Compile features for each trace (component).
    Tau_p_max, tau_c and features from measure_on_trace_data_type.
    
    Parameters
    ----------
    _trace: obspy trace
        Original data.
    Windows: dict
        Picked arrivals.
    src: pandas dataframe
        Source information
    win_len: float
        Length of the data window in seconds        
    window_split: int
        Split the data into increasingly long windows
        
    Returns
    -------    
    measure: dict
        Features extracted for each trace.     
    """
    
    arrival = UTCDateTime(windows["pick_arrival"])

    trace = cut_window(_trace, arrival, win_len)
    print(trace)

    measure = {}
    _v = tau_p_max(trace, time_padded_before = 0.2)
    channel = trace.stats.channel[2]
    measure["%s.tau_p_max" % channel] = _v["tau_p_max"]

    trace_types = make_disp_vel_acc_records(trace)

    measure.update(measure_tau_c(trace_types, window_split))

    for dtype, data in trace_types.items():
        measure.update(measure_on_trace_data_type(data, dtype, window_split))

    return measure


def select_station_components(st, zchan):
    """
    Select the 3-component (Z, E, N) traces with the 
    same channel prefix.

    Parameters
    ----------
    st: obspy stream
        Original data.
    zchan: string
        Z component name.
        
    Returns
    -------    
    st_select: obspy stream
        Three-component data.      
    """
    
    comps = ["Z", "N", "E"]
    st_select = obspy.Stream()
    for comp in comps:
        chan_id = zchan[:-1] + comp
        _st = st.select(id=chan_id)
        if len(_st) == 0:
            continue
        st_select.append(_st[0])

    return st_select


def measure_on_station_stream(src, st, chan_win):
    """
    Make features. Used by measure_on_stream.
    Also add source-station distance, channel name, 
    source time and magnitude to the feature matrix.
    
    Parameters
    ----------
    src: pandas dataframe
        Earthuake information.
    st: obspy stream
        Original data.
    chan_win: dict
        Picked arrivals.
        
    Returns
    -------    
    measure: dict
        Features extracted for each stream. 
             
    """
    
    measure = {}
    for tr in st:
        _m = measure_on_trace(tr, chan_win, src)
        measure.update(_m)
        #print("Number of measurements in trace: %d " % len(_m))

    # add common information
    measure["distance"] = chan_win["distance"]
    measure["channel"] = st[0].id
    measure["source"] = "%s" % UTCDateTime(src.time)
    measure["magnitude"] = src.mag
    return measure


def measure_on_stream(src, waveform_file, window_file):
    """
    Make features for all stations (three components).
    
    Parameters
    ----------
    src: pandas dataframe
        Earthuake information.
    waveform_file: string
        Seismogram file name.
    window_file: string
        File name for picked arrivals (windows).
        .       
    Returns
    -------    
    measure: dict
        Features for all the earthquakes. 
    missing_stations: int
        Number of missing stations.
    """
    
    try:
        st = read_waveform(waveform_file)
    except Exception as err:
        print("Error reading waveform(%s): %s" % (waveform_file, err))
        return

    try:
        windows = load_json(window_file)
    except Exception as err:
        print("Error reading window file: %s" % err)
        return

    results = []
    missing_stations = 0
    for zchan, chan_win in windows.items():
        st_comp = select_station_components(st, zchan)
        _nw = st_comp[0].stats.network
        _sta = st_comp[0].stats.station
        print("-" * 10 + " station: %s.%s " % (_nw, _sta) + "-" * 10)
        if len(st_comp) != 3:
            missing_stations += 1
            continue
        try:
            measure = measure_on_station_stream(src, st_comp, chan_win)
            results.append(measure)
            print("Number of measurements in station stream: %d"
                  % len(measure))
        except Exception as err:
            print("Failed to process data due to: %s" % err)

    return {"measure": results, "missing_stations": missing_stations}


def save_measurements(results, fn):
    """
    Save the feature matrix.
    
    Parameters
    ----------
    results: list
        Feature matrix.
    fn: string
        Name for the saved file.        .       
    """
    
    data = {}
    for k in results[0]:
        data[k] = []

    for d in results:
        for k, v in d.items():
            data[k].append(v)

    df = pd.DataFrame(data)
    print("Save features to file: %s" % fn)
    df.to_csv(fn)


def main():
    """
    Generate the feature matrix (measurements.csv) from the waveform data.
    """
    
    data_path = '../data/'
    sources = pd.read_csv(data_path + "source.csv")
    sources.sort_values("time", ascending=False, inplace=True)
    
    waveform_base = data_path + "preprocessed_data"
    window_base = data_path + "data_windows"
    
    nsources = len(sources)
    results = []
    missing_stations = 0
    for idx in range(nsources):
        src = sources.loc[idx]
        # if src.mag < 3.2:
        #    continue
        origin_time = obspy.UTCDateTime(src.time)
        print("=" * 10 + " [%d/%d]Source(%s, mag=%.2f, dep=%.2f km) "
              % (idx + 1, nsources, origin_time, src.mag, src.depth) +
              "=" * 10)
    
        waveform_file = os.path.join(
            waveform_base, "%s" % origin_time, "CI.mseed")
        window_file = os.path.join(window_base, "%s.json" % origin_time)
    
        _m = measure_on_stream(src, waveform_file, window_file)
        if _m is not None:
            results.extend(_m["measure"])
            missing_stations += _m["missing_stations"]
        print(" *** Missing stations in total: %d ***" % missing_stations)
    
    save_measurements(results, data_path + "measurements.csv")


if __name__ == "__main__":
    main()
