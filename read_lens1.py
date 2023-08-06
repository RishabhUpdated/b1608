from astropy.io import ascii
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np


infile = "lens1_lightcurve.dat"
dat1 = ascii.read(infile, names=['day', 'fluxa', 'fluxb', 'fluxc', 'fluxd', 'erra', 'errb', 'errc', 'errd'])
sigma = 3
clip_range = 1


# Generate evenly-spaced points from given unevenly-spaced data
def interpolate(dat1):
    data = dat1
    time = np.linspace(min(dat1['day']), max(dat1['day']), len(dat1['day']))
    print(time)
    data['day'][0] = time[0]
    data['fluxa'][0] = dat1['fluxa'][0]
    data['fluxb'][0] = dat1['fluxb'][0]
    data['fluxc'][0] = dat1['fluxc'][0]
    data['fluxd'][0] = dat1['fluxd'][0]

    for i in range(1, len(dat1['day']) - 1):
        data['day'][i] = time[i]
        bounding_indeces = getBoundingPointIndeces(data['day'][i], dat1['day'])
        print(data['day'][i])
        print(dat1['day'][bounding_indeces[0]])
        print(dat1['day'][bounding_indeces[1]])
        print()

        m_a = (dat1['fluxa'][bounding_indeces[1]] - dat1['fluxa'][bounding_indeces[0]]) / ((dat1['day'][bounding_indeces[1]] - dat1['day'][bounding_indeces[0]]))
        b_a = dat1['fluxa'][bounding_indeces[1]] - (m_a * dat1['day'][bounding_indeces[1]])
        data['fluxa'][i] = m_a * data['day'][i] + b_a

        m_b = (dat1['fluxb'][bounding_indeces[1]] - dat1['fluxb'][bounding_indeces[0]]) / ((dat1['day'][bounding_indeces[1]] - dat1['day'][bounding_indeces[0]]))
        b_b = dat1['fluxb'][bounding_indeces[1]] - (m_b * dat1['day'][bounding_indeces[1]])
        data['fluxb'][i] = m_b * data['day'][i] + b_b

        m_c = (dat1['fluxc'][bounding_indeces[1]] - dat1['fluxc'][bounding_indeces[0]]) / ((dat1['day'][bounding_indeces[1]] - dat1['day'][bounding_indeces[0]]))
        b_c = dat1['fluxc'][bounding_indeces[1]] - (m_c * dat1['day'][bounding_indeces[1]])
        data['fluxc'][i] = m_c * data['day'][i] + b_c

        m_d = (dat1['fluxd'][bounding_indeces[1]] - dat1['fluxd'][bounding_indeces[0]]) / ((dat1['day'][bounding_indeces[1]] - dat1['day'][bounding_indeces[0]]))
        b_d = dat1['fluxd'][bounding_indeces[1]] - (m_d * dat1['day'][bounding_indeces[1]])
        data['fluxd'][i] = m_d * data['day'][i] + b_d

    data['fluxa'][len(dat1['fluxa']) - 1] = dat1['fluxa'][len(dat1['fluxa']) - 1]
    data['fluxb'][len(dat1['fluxb']) - 1] = dat1['fluxb'][len(dat1['fluxb']) - 1]
    data['fluxc'][len(dat1['fluxc']) - 1] = dat1['fluxc'][len(dat1['fluxc']) - 1]
    data['fluxd'][len(dat1['fluxd']) - 1] = dat1['fluxd'][len(dat1['fluxd']) - 1]

    return data

def getBoundingPointIndeces(d, raw_time):
    points = []
    for i in range(len(raw_time)):
        if (d <= raw_time[i]):
            points.append(i - 1)
            points.append(i)
            break
    
    return points

# Smoothen curve
def apply_gaussian_filter(data, sigma):

    days = data['day']
    fluxa = data['fluxa']
    fluxb = data['fluxb']
    fluxc = data['fluxc']
    fluxd = data['fluxd']

    # Sort the data based on days to ensure correct order
    sorted_indices = np.argsort(days)
    days_sorted = days[sorted_indices]
    fluxa_sorted = fluxa[sorted_indices]
    fluxb_sorted = fluxb[sorted_indices]
    fluxc_sorted = fluxc[sorted_indices]
    fluxd_sorted = fluxd[sorted_indices]
    
    # Calculate the time intervals between data points
    time_intervals = np.diff(days_sorted)
    
    # Calculate the sigma for gaussian filter based on the mean time interval
    mean_time_interval = np.mean(time_intervals)
    sigma_ratio = sigma / mean_time_interval
    
    # Apply the gaussian filter on the fluxes
    data['fluxa'] = gaussian_filter1d(fluxa_sorted, sigma=sigma_ratio, mode='constant')
    data['fluxb'] = gaussian_filter1d(fluxb_sorted, sigma=sigma_ratio, mode='constant')
    data['fluxc'] = gaussian_filter1d(fluxc_sorted, sigma=sigma_ratio, mode='constant')
    data['fluxd'] = gaussian_filter1d(fluxd_sorted, sigma=sigma_ratio, mode='constant')
    
    return data

dat1_int = interpolate(dat1)
dat1_int_gaus = apply_gaussian_filter(dat1_int, sigma=sigma)
for i in range(clip_range):
    dat1_int_gaus.remove_row(0)
    dat1_int_gaus.remove_row(-1)

# Create new time array with horizontal translation
def hshift(delay):
    for i in range(len(dat1_int_gaus['day'])):
        dat1_int_gaus['day'][i] -= delay

    return dat1_int_gaus['day']

# Translate curve vertically
def vshift(curve, shift):
    for i in range(len(dat1_int_gaus['day'])):
        dat1_int_gaus['flux' + curve][i] += shift

# Scale curve vertically
def vscale(curve, factor):
    for i in range(len(dat1_int_gaus['day'])):
        dat1_int_gaus['flux' + curve][i] *= factor

def halign(reference_curve, mobile_curve):
    lengths = []
    passed_center = False
    
    common_range = 1
    average_length = 0

    for s in range(2*len(dat1_int_gaus['day']) - 1):
        for p in range(common_range):
            if (not passed_center):
                average_length += abs(dat1_int_gaus['flux' + reference_curve][p] - dat1_int_gaus['flux' + mobile_curve][len(dat1_int_gaus['day']) - common_range + p])
            else:
                average_length += abs(dat1_int_gaus['flux' + reference_curve][len(dat1_int_gaus['day']) - common_range + p] - dat1_int_gaus['flux' + mobile_curve][p])

        average_length /= common_range

        if (not passed_center):
            lengths.append((common_range - len(dat1_int_gaus), average_length))
            common_range += 1
            if (common_range == len(dat1_int_gaus['day'])):
                passed_center = not passed_center
        else:
            lengths.append((len(dat1_int_gaus) - common_range, average_length))
            common_range -= 1
        
    return (lengths)

# vscale('b', 3.5)
# vshift('b', np.mean(dat1_int_gaus['fluxa']) - np.mean(dat1_int_gaus['fluxb']))
# lengths_ab = halign('b', 'a')
shift_ab = -9

# vscale('c', 4.5)
# vshift('c', np.mean(dat1_int_gaus['fluxa']) - np.mean(dat1_int_gaus['fluxc']) + 0.5) 
# lengths_cb = halign('b', 'c')
shift_cb = -11

# vscale('d', 5.9)
# vshift('c', np.mean(dat1_int_gaus['fluxa']) - np.mean(dat1_int_gaus['fluxd']) - 0.25)
# lengths_db = halign('b', 'd')
shift_db = -21

# plt.errorbar(dat1_int_gaus['day'], dat1_int_gaus['fluxb'], yerr=dat1['errb'], fmt='bp')
# plt.plot(hshift(dat1['day'][9] - dat1['day'][0]), dat1_int_gaus['fluxa'], "co")
# plt.errorbar(dat1_int_gaus['day'], dat1_int_gaus['fluxa'], yerr=dat1['erra'], fmt='cp')
# hshift(dat1['day'][0] - dat1['day'][9])
# plt.plot(hshift(dat1['day'][11] - dat1['day'][0]), dat1_int_gaus['fluxc'], "go")
# plt.errorbar(dat1_int_gaus['day'], dat1_int_gaus['fluxc'], yerr=dat1['errc'], fmt='cp')
# hshift(dat1['day'][0] - dat1['day'][11])
# plt.plot(hshift(dat1['day'][21] - dat1['day'][0]), dat1_int_gaus['fluxd'], "yo")
# plt.errorbar(dat1_int_gaus['day'], dat1_int_gaus['fluxd'], yerr=dat1['errd'], fmt='yp')
# hshift(dat1['day'][0] - dat1['day'][21])
# plt.show()
plt.plot(dat1['day'], dat1['fluxa'], "go")
plt.plot(dat1['day'], dat1['fluxb'], "bo")
plt.plot(dat1['day'], dat1['fluxc'], "yo")
plt.plot(dat1['day'], dat1['fluxd'], "co")
plt.savefig("raw_lens1_data.png")


print("Time delay for curve a with respect to b: {} days".format(shift_ab * (dat1_int_gaus['day'][1] - dat1_int_gaus['day'][0])))
print("Time delay for curve c with respect to b: {} days".format(shift_cb * (dat1_int_gaus['day'][1] - dat1_int_gaus['day'][0])))
print("Time delay for curve d with respect to b: {} days".format(shift_db * (dat1_int_gaus['day'][1] - dat1_int_gaus['day'][0])))
