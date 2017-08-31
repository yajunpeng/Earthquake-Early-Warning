from __future__ import print_function, division
import os
import sys
import numpy as np
import h5py
import pandas as pd
import obspy
from obspy import UTCDateTime
from utils import load_json, read_waveform, make_disp_vel_acc_records

def select_station_components(st, zchan):
    """
    Select the 3-component traces from the same channel.
    
    Parameters
    ----------
    st: Obspy stream
        Waveform data.
    zchan: string 
        Z component name at a certain station.
        
    Returns
    -------    
    st_select: Obspy stream.
        Three-component waveform data.
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


def extract_station_stream(st, 
                           chan_win, 
                           interp_npts, 
                           interp_sampling_rate):
    """
    Extract data at each station. These data are
    interpolated, and then transformed to displacement, 
    velocity and acceleration data.
    
    Parameters
    ----------
    st: Obspy stream
        Waveform data.
    chan_win: dict 
        Window information (e.g. start tiem)
    interp_npts: int
        Number of points per trace of waveform
        after interpolation.
    interp_sampling_rate: int
        Sampling rate after interpolation.        
        
    Returns
    -------    
    st_select: Obspy stream.
        Three-component waveform data.
    """
    arrival = UTCDateTime(chan_win["pick_arrival"])
    data = {}

    for tr in st:
        data[tr.stats.channel[-1]] = \
            make_disp_vel_acc_records(tr, arrival-1.,
                                      interp_flag=True,
                                      interp_npts=interp_npts, 
                                      interp_sampling_rate=interp_sampling_rate)
    return data


def extract_on_stream(src, stations, origin_time,
                      waveform_file, window_file,
                      interp_npts, interp_sampling_rate):
    """
    Extract all the data. 
    
    Parameters
    ----------
    src: pandas dataframe
        Source information (e.g. time,
        location, magnitude)
    stations: pandas dataframe 
        Station information.
    origin_time: UTCDateTime
        Earthquake origin time
    waveform_file: MSEED
        Preprocessed waveform data file.   
    window_file: json
        Data window information.
    interp_npts: int
        Number of points per trace of waveform
        after interpolation.
    interp_sampling_rate: int
        Sampling rate after interpolation.        
        
        
    Returns
    -------    
    Dataset: dict
        "data", "missing_stations",
        "magnitude", "distance" (source-station distance),
        "station_type" (broadband vel or strong motion acc),
        "source" (earthquake origin time)      
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

    dataset = []
    distance = []
    station_type = []
    source = []
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
            print(st_comp)
            station_data = extract_station_stream(
                st_comp, chan_win, interp_npts, interp_sampling_rate)
            if len(station_data) == 3:
                if max(station_data['Z']['disp']) < 1e-2:
                    dataset.append(station_data)
                    distance.append(chan_win["distance"])
                    station_type.append(zchan)
                    source.append(origin_time)
        except Exception as err:
            print("Failed to process data due to: %s" % err)

    if len(distance) != len(dataset):
        raise ValueError("dimension of distance not same as dataset")
    mags = [src.mag, ] * len(dataset)
    print("Number of stations include: %d" % len(dataset))
    return {"data": dataset, "missing_stations": missing_stations,
            "magnitude": mags, "distance": distance,
            "station_type": station_type, "source": source}


def save_data(data, magnitude, distance,
              station_type, source,
              interp_npts, output_path):
    """
    Save data to a csv file.
    """
    print("Number of measurements: %d" % len(data))
    print("Number of magnitudes: %d" % len(magnitude))
    print("Number of distance: %d" % len(distance))
    print("Number of station_type: %d" % len(station_type))
    print("Number of sources: %d" % len(source))
    if len(data) != len(magnitude):
        raise ValueError("Length mismatch between data and magnitude:"
                         "%d, %d" % (len(data), len(magnitude)))

    if len(data) != len(distance):
        raise ValueError("Length mismatch between data and distance:"
                         "%d, %d" % (len(data), len(distance)))

    array = np.zeros([len(data), 9, interp_npts])
    for idx, d in enumerate(data):
        ncol = -1
        for data_type in ['acc', 'vel', 'disp']:
            for comp in ['Z', 'E', 'N']:
                ncol += 1
                array[idx, ncol, :] = d[comp][data_type]

    f = h5py.File(output_path + "dataset_nn.h5", 'w')
    f.create_dataset("waveform", data=array)
    f.create_dataset("magnitude", data=magnitude)
    f.create_dataset("distance", data=distance)
    #f.create_dataset("station_type", data=station_type)
    #f.create_dataset("source", data=source)
    f.close()
    
    df = pd.DataFrame({"station_type": station_type, 
                       "source": source})
    df.to_csv(output_path + "dataset_source_nn.csv")


def main():
    data_path = '../data/'
    sources = pd.read_csv(data_path + "source.csv")
    sources = sources.sort_values("time", ascending=True)
    sources = sources.reset_index()
    
    stations = pd.read_csv(data_path + "station.csv")
    
    waveform_base = data_path + "preprocessed_data"
    window_base = data_path + "data_windows"
    
    interp_npts = 120
    interp_sampling_rate = 20
    
    nsources = len(sources)
    data = []
    magnitude = []
    distance = []
    station_type = []
    source = []
    for idx in range(nsources):
        src = sources.loc[idx]
        origin_time = UTCDateTime(src.time)
        print("=" * 10 + " [%d/%d]Source(%s, mag=%.2f, dep=%.2f km) "
              % (idx + 1, nsources, origin_time, src.mag, src.depth) +
              "=" * 10)
    
        waveform_file = os.path.join(
            waveform_base, "%s" % origin_time, "CI.mseed")
        window_file = os.path.join(window_base, "%s.json" % origin_time)
    
        _d = extract_on_stream(src, stations, src.time,
                               waveform_file, window_file,
                               interp_npts, interp_sampling_rate)
        if _d is not None and len(_d["data"]) > 0:
            data.extend(_d["data"])
            magnitude.extend(_d["magnitude"])
            distance.extend(_d["distance"])
            station_type.extend(_d["station_type"])
            source.extend(_d["source"])
            
    
    save_data(data, magnitude, distance, 
              station_type, source, interp_npts,
              data_path)



if __name__ == "__main__":
    main()
