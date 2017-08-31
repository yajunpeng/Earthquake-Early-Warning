"""
Pick the seismic arrival as the start of the data window.
Method: recursive sta/lta
"""
import os
import sys
import pandas as pd
import obspy
from obspy import UTCDateTime
from arrival_pick import p_wave_onset_and_SNR
#from arrival_pick import plot_stream_and_arrival
from utils import sort_station_by_epicenter_distance, \
    get_predicted_first_arrival, dump_json


def filter_stream_on_distance(st, stations):
    """
    Select data from stations close to the earthquakes.
    
    Parameters
    ----------
    st: obspy stream
        Seismic data.
    stations: pandas dataframe
        Stations close to earthquakes.
        .
    Returns
    -------    
    st_new: obspy stream
        New data.
    dists: list
        Distances from station to earthquake.
    """     
    
    dists = []
    st_new = obspy.Stream()
    idx_trace = 0
    for i in range(len(stations)):
        _nw = stations.loc[i].network
        _sta = stations.loc[i].station
        _dist = stations.loc[i].distance
        _st = st.select(network=_nw, station=_sta)
        if len(_st) == 0:
            continue
        for tr in _st:
            idx_trace += 1
            # print("%d - [%s] -- %.5f" % (idx_trace, tr.id, _dist))
            st_new.append(tr)
            dists.append(_dist)

    return st_new, dists


def filter_and_pick_arrival_on_stream(src, stations, _st, dist_threshold=1.5,
                                      traveltime_threshold=1.0):
    """
    Select useful data and pick arrivals.
    
    Parameters
    ----------
    src: pandas datafram
        Source information.
    stations: pandas dataframe
        Stations information.
    _st: obspy stream
        Seismic data.
    dist_threshold: float
        Maximum distance from station to 
        earthquake (in degrees).
    traveltime_threshold: float
        Maximum difference between theoretical 
        predictions and picked arrivals (
        in seconds.)
        .
    Returns
    -------  
    arrival_info: dict
        {Station info: {Picked arrival time,
                        theoretical arrival time,
                        station-event distance}}
    
    """       
    
    st = _st.select(channel="BHZ")
    st += _st.select(channel="HHZ")
    st += _st.select(channel="HLZ")
    st += _st.select(channel="HNZ")
    
    stations = sort_station_by_epicenter_distance(src, stations)
    stations_nearby = stations.loc[stations["distance"] < dist_threshold]
    print("Number of potential stations(< %.2f degree): %d"
          % (dist_threshold, len(stations_nearby)))

    st, dists = filter_stream_on_distance(st, stations_nearby)
    print("Number of traces available: %d" % len(st))

    picks = {}
    for tr in st:
        _pick = p_wave_onset_and_SNR(
            tr, UTCDateTime(src.time), SNR_threshold=100, SNR_plot_flag=False,
            trigger_plot_flag=False)
        picks[tr.id] = _pick

    n = 0
    for tid, p in picks.items():
        if p["P_pick"] is not None:
            n += 1
    print("Number of valid picks(SNR and hfreq pick): %d" % n)

    arrival_info = {}
    arrivals = get_predicted_first_arrival(src, dists)
    for idx, tr in enumerate(st):
        pick = picks[tr.id]["P_pick"]
        if pick is None:
            continue
        arr = arrivals[idx]
        arr = UTCDateTime(src.time) + arr
        time_diff = pick["time"] - arr

        if abs(time_diff) > traveltime_threshold:
            continue
        arrival_info[tr.id] = {"pick_arrival": "%s" % pick["time"],
                               "theo_arrival": "%s" % arr,
                               "distance": dists[idx]}

    print("Number of arrivals within range(< %.2f sec) of theoretical"
          " arrival time: %d" % (traveltime_threshold, len(arrival_info)))
    # plot_stream_and_arrival(st_new, picks, UTCDateTime(src.time),
    #                        arrivals)
    return arrival_info


def main():
    """
    Select stations that are less than 1.5 degrees
    from the earthquakes. The difference between picked 
    arrivals and the theoretical arrivals is required to 
    be less than 1s. All the useful arrival information is 
    saved in disk.
    """
    
    data_path = '../data/'
    sources = pd.read_csv(data_path + "source.csv")
    sources.sort_values("time", ascending=False, inplace=True)

    stations = pd.read_csv(data_path + "station.csv")

    database = data_path + "preprocessed_data"
    nsources = len(sources)
    outputbase = data_path + "data_windows"
    if not os.path.exists(outputbase):
        os.makedirs(outputbase)
    n_total_pick = 0
    for idx in range(0, nsources):
        src = sources.loc[idx]
        origin_time = obspy.UTCDateTime(src.time)
        print("=" * 5 + " [%d/%d]Source(%s, mag=%.2f, dep=%.2f km) "
              % (idx + 1, nsources, origin_time, src.mag, src.depth) +
              "=" * 5)
        datafile = os.path.join(database, "%s" % origin_time, "CI.mseed")
        if not os.path.exists(datafile):
            continue
        try:
            st = obspy.read(datafile, format="MSEED")
        except Exception as err:
            print("Failed to read in file(%s) due to: %s" % (datafile, err))
            continue
        if src.depth < 0:
            src.depth = 0.5
        #    print("Skip source due to depth < 0")
        #    continue
        arrival_info = filter_and_pick_arrival_on_stream(src, stations, st)
        n_total_pick += len(arrival_info)
        outputfn = os.path.join(
            outputbase, "%s.json" % obspy.UTCDateTime(src.time))
        print("output file: %s" % outputfn)
        dump_json(arrival_info, outputfn)
        print("--- total counts: %d" % n_total_pick)


if __name__ == "__main__":
    main()
