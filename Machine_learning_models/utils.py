from __future__ import print_function, division
import json
from sklearn import preprocessing
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

def load_json(fn):
    with open(fn) as fh:
        return json.load(fh)

def dump_json(content, fn):
    with open(fn, 'w') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)

def print_title(title, symbol="*", slen=20, simple_mode=False):
    total_len = len(title) + 2 + 2 * slen
    if not simple_mode:
        print(symbol * total_len)
    print(symbol * slen + " %s " % title + symbol * slen)
    if not simple_mode:
        print(symbol * total_len)
        
def standarize_feature(X_train, X_test):

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def plot_map(station_lats, station_lons, event_lats, event_lons):
    """
    Plot station and event locations on a map. 
    This makes a better plot than "plot_stations_and_events",
    but the plotting is slower.
    """

    m = Basemap(projection='merc', lon_0=-118, lat_0=34.5,
                llcrnrlon=-121, llcrnrlat=32.25, urcrnrlon=-115, urcrnrlat=36.75,
                resolution='h')
                
    m.drawcoastlines()   
    m.drawmapboundary(fill_color='skyblue')
    m.fillcontinents(color='wheat', lake_color='skyblue')
    #m.etopo()
    #m.bluemarble()
    m.drawmeridians(np.linspace(-122, -114, 5), labels=[0, 0, 1, 1],
                    fmt="%.2f", dashes=[2, 2], fontsize=15)
    m.drawparallels(np.linspace(32, 38, 5), labels=[1, 1, 0, 0],
                    fmt="%.2f", dashes=[2, 2], fontsize=15)
    #m.etopo()
    x_station, y_station = m(station_lons, station_lats)
    m.scatter(x_station, y_station, 50, color="r", marker="v",
              edgecolor="k", zorder=3)
    x_event, y_event = m(event_lons, event_lats)
    m.scatter(x_event, y_event, 10, color="k", marker=".",
              zorder=3)

def plot_map_us():
    # setup Lambert Conformal basemap.
    # set resolution=None to skip processing of boundary datasets.
    m = Basemap(width=12000000,height=9000000,projection='lcc',
                resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
    m.etopo()
    
        
def plot_stations_and_events(station_loc, catalog, map_flag = False):    
    if map_flag == False:
        plt.plot(station_loc.longitude, station_loc.latitude, 'r^')
        plt.plot(catalog.longitude, catalog.latitude, '.') 
        plt.xlabel('Longitude (degrees)')
        plt.ylabel('latitdue (degrees)')
    else:
        plt.figure(figsize=(10, 10))
        plot_map(station_loc.latitude.values, station_loc.longitude.values,
                 catalog.latitude.values, catalog.longitude.values)
        plt.annotate("LA", xy=(0.33, 0.33), xycoords='axes fraction', fontsize = 40)
        
    
        