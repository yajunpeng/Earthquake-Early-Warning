from __future__ import print_function, division
import os
import time
import glob
import obspy

def select_by_sampling_rate(st, threshold=15):
    """
    Select data with sampling rate above threshold.
    
    Parameters
    ----------    
    st: obspy stream
        seismic data
    threshold: int
        sampling rate threshold
    
    Returns
    ------- 
    st_new: obspy stream
        new seismic stream data        
    """    
    st_new = obspy.Stream()
    for tr in st:
        if tr.stats.sampling_rate < threshold:
            continue
        st_new.append(tr)

    print("Number of traces change(>%d Hz): %d --> %d"
          % (threshold, len(st), len(st_new)))
    return st_new


def filter_waveform(st, freq=0.075, taper_percentage=0.05, taper_type="hann"):
    """
    Apply highpass filter to seismogram.
    
    Parameters
    ----------    
    st: obspy stream
        Seismic data
    freq: float
        Frequency cutoff
    taper_percentage: float
        Taper a small segment at the start and
        the end of the waveform
    
    Returns
    ------- 
    st: obspy stream
        Filtered seismic data     
    """    
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=taper_percentage, type=taper_type)

    for tr in st:
        tr.filter("highpass", freq=freq)

    return st


def detrend_waveform(st):
    """
    Detrend the waveform. Typically applied before filter.
    
    Parameters
    ----------    
    st: obspy stream
        Seismic data
    
    Returns
    ------- 
    st: obspy stream
        Detrended seismic data     
    """      
    st.detrend("linear")
    st.detrend("demean")

    return st


def remove_instrument_gain(st, invs):
    """
    Correct for the ground motion amplitude by removing intrument gain.
    
    Parameters
    ----------    
    st: obspy stream
        Seismic data
    invs: dict 
        obspy inventories
    
    Returns
    ------- 
    st_new: obspy stream
        Corrected data.   
    """      
    ntr1 = len(st)
    n_missing_inv = 0
    print("Number of inventories: %d" % len(invs))

    st_new = obspy.Stream()
    for tr in st:
        nw = tr.stats.network
        sta = tr.stats.station
        tag = "%s_%s" % (nw, sta)
        if tag not in invs:
            # print("Missing station: %s" % tag)
            n_missing_inv += 1
            continue
        inv = invs[tag].select(
            network=nw, station=sta, location=tr.stats.location,
            channel=tr.stats.channel,
            starttime=tr.stats.starttime, endtime=tr.stats.endtime)
        if len(inv) == 0:
            # print("Missing inventory: %s" % tr.id)
            continue
        try:
            sens = inv[0][0][0].response.instrument_sensitivity.value
            tr.data /= sens
            st_new.append(tr)
        except Exception as err:
            print("Error remove instrument gain(%s): %s" % (tr.id, err))

    ntr2 = len(st_new)
    print("Number of traces missing inventory: %d" % n_missing_inv)
    print("Number of trace change after remove instrument gain: %d --> %d"
          % (ntr1, ntr2))
    return st_new


def process_one_file(fn, invs, outputfn):
    """
    Preprocess one trace of data.
    
    Parameters
    ----------    
    fn: string
        File name.
    invs: dict 
        obspy inventories
    outputfn: string
        Output directory
  
    """         
    print("-" * 10 + "Process file: %s" % fn, "-" * 10)
    st = obspy.read(fn)
    st = select_by_sampling_rate(st)
    #st = filter_waveform(st)
    st = detrend_waveform(st)
    # instrument gain should be removed to obtain the true amplitude
    st = remove_instrument_gain(st, invs)

    print("Saved streams to file: %s" % outputfn)
    st.write(outputfn, format="MSEED")


def process_one_event(dirname, invs, outputdir):
    """
    Preprocess data for one earthquake.
    
    Parameters
    ----------    
    dirname: string
        Data directory.
    invs: dict
        obspy inventories
    outputfn: string
        Output directory
  
    """         
    print("outputdir: %s" % outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    fn = os.path.join(dirname, "CI_*.mseed")
    if os.path.exists(fn):
        outputfn = os.path.join(outputdir, "CI.mseed")
        try:
            process_one_file(fn, invs, outputfn)
        except Exception as err:
            print("Error processing file(%s) due to: %s" % (fn, err))

    fn = os.path.join(dirname, "AZ_*.mseed")
    if os.path.exists(fn):
        outputfn = os.path.join(outputdir, "AZ.mseed")
        try:
            process_one_file(fn, invs, outputfn)
        except Exception as err:
            print("Error processing file(%s) due to: %s" % (fn, err))


def read_all_inventories():
    """
    Read the obspy inventories using station xml files.
    Returns
    ------- 
    invs: dict
        Obspy inventories.   
    """          
    files = glob.glob("../data/station_xml/*.xml")
    nfiles = len(files)

    invs = {}
    t1 = time.time()
    for idx, f in enumerate(files):
        _t1 = time.time()
        inv = obspy.read_inventory(f)
        _t2 = time.time()
        tag = os.path.basename(f).split(".")[0]
        invs[tag] = inv
        print("[%d/%d]Read staxml(%s): %.2f sec" %
              (idx+1, nfiles, tag, _t2 - _t1))
    t2 = time.time()

    print("Time reading inventories: %.2f sec" % (t2 - t1))
    return invs


def main():
    """
    Waveform data with sampling rate >= 15Hz are selected.
    Then they are detrended and demeaned.
    Instrument gain (recorded in station xml files) are removed 
    to recover the true amplitude.
    """
    
    invs = read_all_inventories()
    # invs = {"CI_ADO": obspy.read_inventory("./station_xml/CI_ADO.xml")}
    data_path = '../data/'        
    outputbase = data_path + "preprocessed_data"
    if not os.path.exists(outputbase):
        os.makedirs(outputbase)

    dirnames = glob.glob(data_path + "raw_data/*")
    ndirs = len(dirnames)
    print("Number of dirs: %d" % ndirs)
    dirnames = sorted(dirnames, reverse=True)
    for idx, _dir in enumerate(dirnames):
        print("=" * 15 + " [%d/%d] dir: %s " % (idx+1, ndirs, _dir) + "=" * 15)
        outputdir = os.path.join(outputbase, os.path.basename(_dir))
        process_one_event(_dir, invs, outputdir)


if __name__ == "__main__":
    main()
