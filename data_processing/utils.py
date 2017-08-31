import json
import obspy
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel


def highpass_filter_waveform(trace, freq):
    """
    Apply highpass filter. Often used when integrate 
    the waveform to avoid low frequency noise.
    
    Parameters
    ----------    
    trace: obspy trace
        seismic data
    freq: float
        Cutoff frequency      
    """     
    trace.detrend('linear')
    trace.taper(0.05)
    trace.filter('highpass', freq=freq, corners=4, zerophase=False)
    # trace.detrend('linear')
    # trace.taper(0.05)


def lowpass_filter_waveform(trace, freq):
    """
    Apply lowpass filter.
    
    Parameters
    ----------    
    trace: obspy trace
        seismic data
    freq: float
        Cutoff frequency      
    """      
    trace.detrend('linear')
    trace.taper(0.05)
    trace.filter('lowpass', freq=freq, corners=4, zerophase=False)
    # trace.detrend('linear')
    # trace.taper(0.05)


def sort_station_by_epicenter_distance(src, stations):
    """
    Sort the stations based on event-station distance.
    Used for remove far away stations.
    
    Parameters
    ----------    
    src: pandas dataframe
        Earthquake information.
    stations: pandas dataframe
        Station information
    
    Returns
    ------- 
    stations: pandas dataframe
        Sorted station info.     
    """     
    
    dists = []
    for idx in range(len(stations)):
        _d = locations2degrees(
            src["latitude"], src["longitude"],
            stations.loc[idx].latitude, stations.loc[idx].longitude)
        dists.append(_d)

    stations["distance"] = dists
    stations = stations.sort_values(by=["distance"])
    stations = stations.reset_index(drop=True)
    return stations


def filter_stream_by_sampling_rate(stream, threshold=19):
    """
    Remove stations with low sampling rate. They are
    generally not seismic stations.
    
    Parameters
    ----------    
    stream: obspy stream
        Seismic data
    threshold: int
        Sampling rate cutoff.
    
    Returns
    ------- 
    stream_filter: obspy stream    
    """       
    stream_filter = obspy.Stream()
    for tr in stream:
        if tr.stats.sampling_rate < threshold:
            continue
        stream_filter.append(tr)

    print("Number of traces change: %d --> %d"
          % (len(stream), len(stream_filter)))
    return stream_filter


def get_predicted_first_arrival(src, dists):
    """
    Predict arrival time using the software
    Tau_p.
    
    Parameters
    ----------    
    src: pandas dataframe
        Earthquake information
    
    Returns
    ------- 
    arrivals: list
        Predicted times.    
    """      
    print("source depth: %.2f km" % src.depth)
    model = TauPyModel(model="prem")
    arrivals = []
    for deg in dists:
        arrivs = model.get_travel_times(
            src.depth, deg, phase_list=("p", "P", "Pn"))
        arrivals.append(arrivs[0].time)
    return arrivals


def load_json(fn):
    """
    Load json file.
    
    Parameters
    ----------    
    fn: string
        File name.
    
    Returns
    ------- 
    Loaded file.   
    """      
    with open(fn) as fh:
        return json.load(fh)


def dump_json(content, fn):
    """
    Save json file.
    
    Parameters
    ----------    
    fn: string
        File name. 
    """          
    with open(fn, 'w') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)
