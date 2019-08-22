import numpy as np
from scipy import signal
import obspy
from obspy.core.utcdatetime import UTCDateTime


### Cross-corelation
def lag_finder(y1, y2, channel=0):
    '''
    Takes 2 ObsPy objects (Trace or Stream) and
    returns delay and correlation matrix between them
    '''
    # Check if Object is a Stream
    if type(y1) != obspy.core.stream.Trace:
        # Extract trace
        y1 = y1[channel]
        y2 = y2[channel]

    n = len(y1)  # Length of the Trace
    sr = int(y1.stats.sampling_rate)  # Sampling rate
    # Compute cross-correlation matrix
    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(
        signal.correlate(y1, y1, mode='same')[int(n / 2)] * signal.correlate(y2, y2, mode='same')[int(n / 2)])
    # Construct an array of indexes
    delay_arr = np.linspace(-0.5 * n / sr, 0.5 * n / sr, n)
    # Get maximum delay
    delay = delay_arr[np.argmax(corr)]
    return delay, corr


def calc_t_est(indexes, delta_t):
    '''
    Takes station names (indexes) and computed delta T;
    returns centered delay times.

    Based on J. C. VanDecar, R. S. Crosson; Determination of teleseismic relative
    phase arrival times using multi-channel cross-correlation and least squares.
    Bulletin of the Seismological Society of America ; 80 (1): 150–169
    '''
    n = len(np.unique(indexes))  # Get number of unique traces
    delta_t = np.append(delta_t, 0)  # Append 0
    # Construct matrix of delta T for each unqiue station pair
    array = (np.asarray([[i[0][-1] for i in indexes]], dtype='object') +
             np.asarray([[i[1][-1] for i in indexes]], dtype='object'))[0]
    # Create matrix of coefficients as in paper
    coef = np.zeros((int(n * (n - 1) / 2 + 1), n))
    coef[[i for i in range(0, len(coef) - 1)], [int(j[0]) - 1 for j in array]] = 1
    coef[[i for i in range(0, len(coef) - 1)], [int(j[1]) - 1 for j in array]] = -1
    coef[-1, :] = 1
    return np.asarray((1 / n) * (np.asmatrix(coef).T) * np.asmatrix(delta_t).T)


### Help functions
def get_coordinates(st, inv_ff):
    '''
    Takes a Stream object and Inventory object,
    returns latitude and lontitude for the Station
    '''
    coordinates = inv_ff.get_coordinates(st[0].stats.network + '.' + st[0].stats.station + '..' + st[0].stats.channel)
    lat = coordinates['latitude']
    lon = coordinates['longitude']
    return lat, lon


def great_circle_dist(lat1, lat2, lon1, lon2):
    '''
    Calculates Great Circle Distance between 2 stations.
    Takes coordinate pairs and returns distance between them.
    Based on: https://en.wikipedia.org/wiki/Great-circle_distance
    '''
    lat1, lat2, lon1, lon2 = np.deg2rad([lat1, lat2, lon1, lon2])  # Convert degrees to radians
    d_sigma = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(
        lon2 - lon1))  # Calculates central angle between them
    return 6371 * 10 ** 3 * d_sigma
