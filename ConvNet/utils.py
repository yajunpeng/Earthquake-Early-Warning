import json
import obspy

def load_json(fn):
    """
    Load json file.
    
    Parameters
    ----------
    fn: string
        File name.
        
    Returns
    -------    
    Dict-like data.
    """
    with open(fn) as fh:
        return json.load(fh)

def dump_json(data, fn):
    """
    Save json file.
    
    Parameters
    ----------
    data: dict-like
    
    fn: string
        File name.
    """
    with open(fn, 'w') as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def read_waveform(fn):
    """
    Read seismic waveform data from MSEED file.
    
    Parameters
    ----------
    fn: string
        File name.
        
    Returns
    -------    
    Waveform: Obspy stream.
    """    
    st = obspy.read(fn, format="MSEED")
    return st

def highpass_filter_waveform(trace, freq):
    """
    Highpass filter data.
    
    Parameters
    ----------
    trace: obspy trace
        Single trace of waveform.
    freq: float
        Highpass frequency.
    """   
    trace.detrend('linear')
    trace.taper(0.05)
    trace.filter('highpass', freq=freq, corners=4, zerophase=False)
    # trace.detrend('linear')
    # trace.taper(0.05)


def lowpass_filter_waveform(trace, freq):
    """
    Lowpass filter data.
    
    Parameters
    ----------
    trace: obspy trace
        Single trace of waveform.
    freq: float
        Lowpass frequency.
    """  
    trace.detrend('linear')
    trace.taper(0.05)
    trace.filter('lowpass', freq=freq, corners=4, zerophase=False)
    # trace.detrend('linear')
    # trace.taper(0.05)

def differentiate_waveform(_trace):
    """
    Convert velocity to acceleration.
    
    Parameters
    ----------
    _trace: obspy trace
        Single trace of waveform.
        
    Returns
    -------    
    trace: obspy trace
        Acceleration waveform.
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
        Single trace of waveform.
        
    Returns
    -------    
    trace: obspy trace
        Velocity or displacement waveform.
    """
    trace = _trace.copy()
    trace.integrate()
    highpass_filter_waveform(trace, 0.075)
    return trace


def make_disp_vel_acc_records(trace, arrival,
                              interp_flag=True,
                              interp_npts=400, 
                              interp_sampling_rate=20):
    """
    Several empircal measurements for earthquake early warning
    from literature (tau_p_max, tau_c, P_d, P_v, P_a)

    Parameters
    ----------
    trace: obspy trace
        Single trace of waveform.
    arrival: UTCDateTime
        Seismic wave arrival time.
    interp_flag: boolean
        If true, interpolate the waveform.
    interp_npts: int
        Number of data points per waveform
        after interpolation.
    interp_sampling_rate: int
        Sampling rate after interpolation.
    
        
    Returns
    -------    
    Acceleration, velocity and displacement data: Dict
    """
    station_type = trace.stats.channel[:2]
    #arrival = UTCDateTime(chan_win["pick_arrival"])
    
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
        
    trace_acc.interpolate(interp_sampling_rate, starttime=arrival,
               npts=interp_npts)
    trace_vel.interpolate(interp_sampling_rate, starttime=arrival,
               npts=interp_npts)
    trace_disp.interpolate(interp_sampling_rate, starttime=arrival,
               npts=interp_npts)    
    
    return {"acc": trace_acc.data, "vel": trace_vel.data, "disp": trace_disp.data}