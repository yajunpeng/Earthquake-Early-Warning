"""
Auxilary functions for extracting features from
seismic data.
"""
import numpy as np
from utils import highpass_filter_waveform, lowpass_filter_waveform
import obspy.signal


def norm(data, order):
    """
    Compute norm of the data for feature extraction.  
    
    Parameters
    ----------
    data: array_like
        Waveforms
    order: int
        
    Returns
    -------    
    norm of the data: float   
    """       
    return np.linalg.norm(data, ord=order)


def envelope(data):
    """
    Compute envolope of the waveform for feature extraction.   
    
    Parameters
    ----------
    data: array_like
        Waveforms
        
    Returns
    -------    
    envelope of the data: array-like   
    """
    return obspy.signal.filter.envelope(data)


def split_data(data, n=6):
    """
    Split data into n segements.
    
    Parameters
    ----------
    data: array_like
        Waveforms
    n: int
        Number of segments.
        
    Returns
    -------    
    data_split: list of array-like
        Data segments     
    """
    
    npts = len(data)
    npts_split = int(npts / n)

    data_split = []
    for i in range(n):
        _d = data[i*npts_split: (i+1)*npts_split]
        data_split.append(_d)

    return data_split


def split_accumul_data(data, n=6):
    """
    Split data into n increasingly long segements
    (each from the start of the data).
    
    Parameters
    ----------
    data: array_like
        Waveforms
    n: int
        Number of segments.
        
    Returns
    -------    
    data_split: list of array-like
        Data segments     
    """    
    
    npts = len(data)
    npts_split = int(npts / n)

    data_split = []
    for i in range(n):
        _d = data[0: (i+1)*npts_split]
        data_split.append(_d)

    return data_split


def cut_window(trace, starttime, win_length):
    """
    Cut out the first win_length seconds of P wave
    (Note: for stations close to the event, S wave
    may be included.)
    
    Parameters
    ----------
    trace: obspy trace
        Seimic data
    starttime: obspy UTCDateTime
        The arrival time (start of the data).
    win_length: float
        Length of the data window in seconds.
        
    Returns
    -------    
    trace_cut: obspy trace
    """
    endtime = starttime + win_length
    trace_cut = trace.slice(starttime=starttime, endtime=endtime)
    return trace_cut


def differentiate_waveform(_trace):
    """
    Convert velocity to acceleration
    
    Parameters
    ----------
    _trace: obspy trace
        Seimic data
        
    Returns
    -------    
    trace: obspy trace    
        Differentiated data.
    """
    trace = _trace.copy()
    trace.differentiate()
    highpass_filter_waveform(trace, 0.075)
    return trace


def integrate_waveform(_trace):
    """
    Convert acceleration to velocity,
    or velocity to displacement
    
    Parameters
    ----------
    _trace: obspy trace
        Seimic data
        
    Returns
    -------    
    trace: obspy trace    
        Integrated data.    
    """
    trace = _trace.copy()
    trace.integrate()
    highpass_filter_waveform(trace, 0.075)
    return trace


def make_disp_vel_acc_records(trace):
    """
    Several empircal measurements for earthquake early warning
    from literature (tau_p_max, tau_c, P_d, P_v, P_a)
    
    Parameters
    ----------
    trace: obspy trace
        Seimic data
        
    Returns
    -------    
    acc, vel, disp records: dict of obspy traces    
    """
    station_type = trace.stats.channel[:2]
    if (station_type == 'BH') | (station_type == 'HH'):
        trace_vel = trace.copy()
        trace_disp = integrate_waveform(trace_vel)
        trace_acc = differentiate_waveform(trace_vel)
        highpass_filter_waveform(trace_vel, 0.075)
    elif (station_type == 'HN') | (station_type == 'HL'):
        trace_acc = trace.copy()
        trace_vel = integrate_waveform(trace_acc)
        trace_disp = integrate_waveform(trace_vel)
        highpass_filter_waveform(trace_acc, 0.075)
    else:
        raise ValueError(
            'Wrong instrument! Choose from BH*, HH*, HN*, or HL*.')
    return {"acc": trace_acc, "vel": trace_vel, "disp": trace_disp}


def tau_p_max(_trace, time_padded_before=0, lowpass_for_tau_p=True,
              lowpass_freq=3.0):
    """
    An empirical estimate for the predominant frequency:
    tau_p_i = 2 * pi * sqrt(X_i/D_i)
    X_i = alpha * X_(i-1) + x_i^2
    D_i = alpha * D_(i-1) + ((dx/dt)_i)^2
    x_i is velocity.
    Olson and Allen (2005) used 3 Hz lowpass filter.
    Previous studies show that this measurement seems to be not very robust.
    
    Parameters
    ----------
    _trace: obspy trace
        Seimic data
    time_padded_before: float
        Data added before the picked arrival.
    lowpass_for_tau_p: boolean
        Apply lowpass filter.
    lowpass_freq: float
        Threshold frequency for lowpass filter.
        Not useful when lowpass_for_tau_p is False.
    
    Returns
    -------   
    Info for tau_p: dict
        1) tau_p_max: float   
            The maximum tau_p value
        2) tau_p: array-like
            The tau_p time series.
    """
    trace = _trace.copy()
    df = trace.stats.sampling_rate
    trace_len = len(trace)
    if lowpass_for_tau_p:
        lowpass_filter_waveform(trace, lowpass_freq)

    _trace_types = make_disp_vel_acc_records(trace)
    trace_vel = _trace_types["vel"]
    trace_acc = _trace_types["acc"]

    alpha = 1. - 1. / df
    tau_p = np.zeros(trace_len, )
    X_i = np.zeros(trace_len)
    D_i = np.zeros(trace_len)
    for i_sample in range(trace_len):
        if i_sample == 0:
            X_i[i_sample] = trace_vel[i_sample] ** 2
            D_i[i_sample] = trace_acc[i_sample] ** 2
        else:
            X_i[i_sample] = \
                X_i[i_sample - 1] * alpha + trace_vel[i_sample] ** 2
            D_i[i_sample] = \
                D_i[i_sample - 1] * alpha + trace_acc[i_sample] ** 2
    tau_p = 2 * np.pi * np.sqrt(X_i / D_i)
    tau_p[np.isnan(tau_p)] = 0
    tau_p_max = max(tau_p[int(time_padded_before * df):])

    return {"tau_p_max": tau_p_max, "tau_p": tau_p}


def tau_c(disp, vel):
    """
    Tau_c is a period parameter widely used in EEW literature. 
    If the waveform is monochromatic, this is essentially the period.
    Pd, Pv, Pa are the peak amplitude of displacement, velcoity,
    acceleration within the first a few seconds.
    
    Parameters
    ----------
    disp: array-like
        Displacement record.
    vel: array-like
        Velocity record.        
    
    Returns
    -------   
    tau_c: float
    """
    tau_c = 2 * np.pi * np.sqrt(
        np.sum(disp ** 2) / np.sum(vel ** 2))
    return tau_c
