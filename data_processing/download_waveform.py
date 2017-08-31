"""
Data are publicly available in IRIS data center.
Obspy is used for downloading.
"""
import os
import time
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

def _parse_station_id(station_id):
    """
    Parse a string containing station info.
    
    Parameters
    ----------    
    station_id: string
        station name
    
    Returns
    ------- 
    nw, sta, loc, comp: string
        station network name, station name, 
        station location, component name        
    """
    content = station_id.split("_")
    if len(content) == 2:
        nw, sta = content
        loc = "*"
        comp = "*"
    elif len(content) == 4:
        nw, sta, loc, comp = content
    else:
        raise ValueError("Can't not parse station_id: %s" % station_id)
    return nw, sta, loc, comp

def download_waveform(stations, starttime, endtime, outputdir=None,
                      client=None):
    """
    download wavefrom data from IRIS data center
    
    Parameters
    ----------    
    stations: list of stations, should be list of station ids,
        for example, "II.AAK.00.BHZ". Parts could be replaced by "*",
        for example, "II.AAK.*.BH*"    
    starttime/endtime: obspy UTCDateTime
        start/end of the waveform
    
    Returns
    ------- 
    seismogram data: dict
        {obspy stream object, error code}    
    
    """
    
    if client is None:
        client = Client("IRIS")

    if starttime > endtime:
        raise ValueError("Starttime(%s) is larger than endtime(%s)"
                         % (starttime, endtime))

    if not os.path.exists(outputdir):
        raise ValueError("Outputdir not exists: %s" % outputdir)

    _status = {}
    for station_id in stations:
        error_code = "None"
        network, station, location, channel = _parse_station_id(station_id)

        if outputdir is not None:
            filename = os.path.join(outputdir, "%s.mseed" % station_id)
            if os.path.exists(filename):
                os.remove(filename)
        else:
            filename = None

        try:
            st = client.get_waveforms(
                network=network, station=station, location=location,
                channel=channel, starttime=starttime, endtime=endtime)
            if len(st) == 0:
                error_code = "stream empty"
            if filename is not None and len(st) > 0:
                st.write(filename, format="MSEED")
        except Exception as e:
            error_code = "Failed to download waveform '%s' due to: %s" \
                % (station_id, str(e))
            print(error_code)

        _status[station_id] = error_code

    return {"stream": st, "status": _status}


def check_no_dup(times):
    """
    Remove duplicate events
    
    Parameters
    ----------    
    times: obspy UTCDateTime
        event occurrence times
          
    """    
    tset = set()
    for t in times:
        if t in tset:
            print("Duplicate time: %s" % t)
        tset.add(t)


def stats_stream(streams):
    """
    Show infomation of stations and data traces
    
    Parameters
    ----------    
    streams:
        obpsy object containing the seismic data     
    """    
    
    for sta_id, st in streams.iteritems():
        if st is None:
            print("[%s] is None" % sta_id)
            continue

        stations = set()
        for tr in st:
            tag = "%s.%s" % (tr.stats.network, tr.stats.station)
            stations.add(tag)
        print("[%s]Number of stations and traces: %d, %d"
              % (sta_id, len(stations), len(st)))


def download_func(eventtime, outputbase, before=-60, after=120):
    """
    Download data.
    
    Parameters
    ----------    
    eventtime: obspy UTCDateTime
        earthquake occurrence time
    outputbase: 
        output directory
    before/after: int, float
        time (in seconds) before/after eventtime 
     
    """    
    t1 = time.time()

    outputdir = os.path.join(outputbase, "%s" % eventtime)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    station_list = ["CI_*", "AZ_*"]
    print("Download time: %f, %f" % (before, after))
    print("Station list: %s" % station_list)
    _data = download_waveform(
        station_list, eventtime + before, eventtime + after,
        outputdir=outputdir)
    stats_stream(_data["stream"])

    t2 = time.time()
    print("outputdir: %s -- time: %.2f sec" % (outputdir, t2 - t1))


def check_download_finish(datadir):
    """
    Check downloaded data.
    
    Parameters
    ----------    
    datadir: 
        data directory
    
    Returns
    ------- 
    data existence: boolean        
    """    
    fn = os.path.join(datadir, "CI_*.mseed")
    if os.path.exists(fn):
        return True
    else:
        return False


def main():
    """
    Download data for each earthquake from the catalog (source.csv).
    """
    data_path = '../data/'
    outputbase = data_path + "raw_data"
    if not os.path.exists(outputbase):
        os.makedirs(outputbase)

    sources = pd.read_csv(data_path + "source.csv")
    check_no_dup(sources.time)

    nevents = len(sources)
    print("Number of sources: %d" % nevents)
    for idx, ev in sources.iterrows():
        if idx != (nevents - 1):
            nextdir = os.path.join(
                outputbase, "%s" % UTCDateTime(sources.loc[idx+1].time))
            if check_download_finish(nextdir):
                continue

        t = UTCDateTime(ev.time)
        print(" ===== [idx: %d/%d] -- time: %s ===== " % (idx, nevents, t))
        download_func(t, outputbase)


if __name__ == "__main__":
    main()
