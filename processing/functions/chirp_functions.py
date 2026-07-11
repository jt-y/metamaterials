#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:25:35 2019

@author: radu
"""

import os
import pickle
import glob
import numpy as np
import scipy
from scipy import optimize
import math
import tqdm
from matplotlib import pyplot as plt

def load_data(data_dir, sample_width = 40, sample_offset = 0, title = 'abfs_'):
    fnames = list(sorted(glob.glob(os.path.join(data_dir, "*.pkl"))))
    
    print('Found %s records' % len(fnames))
    # Load into a list of tuples of xmin, xmax, y, data
    data = []
    freqs = []
    avgs = []
    
    for fname in fnames:
        with open(fname, 'rb') as f:
            fft_data = pickle.load(f)
        
        #print(len(fft_data[0]))
        
        
        # Isolate the frequency. In our case, 28kHz is usually around sample 8000 to 10000
        amplitudes = []
        for i in range(len(fft_data)):
            arr = np.array(fft_data[i][sample_offset:(sample_offset + sample_width)])
            
            # JUST GETS THE MAXIMUM
            amplitudes.append(arr.max())               
        
        name = os.path.basename(fname).replace('.pkl', '').replace(title, '')
        freq = [float(coord) for coord in name.split('_')]
        
        freqs.append(freq)
        avgs.append(sum(amplitudes) / len(amplitudes))
        data.append(amplitudes)
        
    if not data:
        raise RuntimeError('No Data Found')  
        
    data = np.array(data)
    
    return data, freqs, avgs


def avg_over_data(data, samples):
    avgdata=[]
    for index in range(len(data)):
        values = data[index]
        avg_values = []        
        for j in range(samples, len(values) - samples):
            avg_values.append(sum(values[j - samples:j + samples]) / (2.0 * samples))        
        avgdata.append(np.array(avg_values))
    return avgdata

def pad_data(data, factor):
    arr = []
    for startp in range(len(data) - 1):
        dp1 = data[startp]
        dp2 = data[startp + 1]

        try:
            dp1 = dp1[0]
            dp2 = dp2[0]
        except:
            dp1 = dp1
            dp2 = dp2
            
        for i in range(factor):
            arr.append((dp1 * (factor - i) + dp2 * i) / factor)
    for i in range(factor):
        try:
            arr.append(data[-1][0]) 
        except:
            arr.append(data[-1]) 
          
    return np.array(arr)
    
def preprocess_chirp(setno, data_dir, frequencies, targetfile, title = 'scppos_'):
    fnames = list(sorted(glob.glob(os.path.join(data_dir, "*.pkl"))))
    
    print('Found %s records' % len(fnames))
    
    data = [[] for _ in frequencies]
    
    for fname in fnames:
        with open(fname, 'rb') as f:
            chirp_data = pickle.load(f)
               
        x, y = [float(coord) for coord in os.path.basename(fname).replace('.pkl', '').replace(title, '').split('_')]
        
        for point in chirp_data:
            freq_index = np.where(frequencies == point[0])[0][0] 
            data[freq_index].append((x, y, point[1], point[2]))
                
    if not data:
        raise RuntimeError('No Data Found')  
        
    for freq in frequencies:
        index = np.where(frequencies == freq)[0][0]       
        np.array(data[index]).dump(targetfile + '_' + str(freq) + '_' + str(setno) + '.pkl')
    
    print('Preprocessing complete!')
    


def combine_multiple_measurements(data_sets, path = "../../data/processed/", avgsize = 1,
                                  average_data = False, figsize = (10, 5), imname = "unknown",
                                  point_start = None, point_end = None, savefigs = False, 
                                  showplot = True, title = "Untitled Plot",
                                  xlabel = "Frequency (Hz)",
                                  ylabel = "Response (dBV)"):
    
    frequencies = []
    ivalues = []
    
    for name in data_sets:
         with open(path + name[0] + "-f.pkl", 'rb') as f:
            frequencies.append(pickle.load(f))
         with open(path + name[0] + "-v.pkl", 'rb') as f:
            ivalues.append(np.array(pickle.load(f)) * name[1])
    
    comb_arr = []
    
    for index in range(len(data_sets)):
        for i in range(len(frequencies[index])):
            comb_arr.append((frequencies[index][i], ivalues[index][i])) 
    
    comb_arr = np.array(comb_arr, dtype = [('freq', float), ('trans', float)])
    comb_arr = np.sort(comb_arr, order = 'freq')

    car2 = []
    
    index = 0
    while (index < len(comb_arr)):
        fin = index
        while fin + 1 < len(comb_arr): 
            if comb_arr['freq'][fin + 1] == comb_arr['freq'][index]:
                fin += 1
            else:
                break            
        car2.append((comb_arr['freq'][index], sum(comb_arr['trans'][index:fin+1]) / (fin - index + 1.0)))
        index = fin + 1
        
    comb_arr = np.array(car2, dtype = [('freq', float), ('trans', float)])
    comb_arr = np.sort(comb_arr, order = 'freq')
    
    if average_data:
        values = avg_over_data([comb_arr['trans']], avgsize)[0]
        frequencies = comb_arr['freq'][avgsize:-avgsize]
    else:
        values = comb_arr['trans']
        frequencies = comb_arr['freq']
    #####

    plt.clf()
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if not point_start:
        point_start = 0
    if not point_end:
        point_end = len(values)
    
    plt.plot(frequencies[point_start:point_end], values[point_start:point_end])
    if savefigs:
        plt.savefig("../../data/images/" + imname + ".png", dpi = 300)
        
    if showplot:
        plt.show()

###############################################################################
        
def plot_transmittance_measurements(data_sets, path = "../../data/processed/transmittance/", 
                                    avgsize = 1, average_data = False, point_start = None, 
                                    point_end = None, y_offset = 0.0):
    
    frequencies = []
    transmittances = []
    
    for name in data_sets:
         with open(path + name[0] + "-f.pkl", 'rb') as f:
            frequencies.append(pickle.load(f))
         with open(path + name[0] + "-v.pkl", 'rb') as f:
            transmittances.append(np.array(pickle.load(f)) * name[1])

    #####
    
    comb_arr = []
    
    for index in range(len(data_sets)):
        for i in range(len(frequencies[index])):
            comb_arr.append((frequencies[index][i], transmittances[index][i])) 
    
    comb_arr = np.array(comb_arr, dtype = [('freq', float), ('trans', float)])
    comb_arr = np.sort(comb_arr, order = 'freq')

    car2 = []
    
    index = 0
    while (index < len(comb_arr)):
        fin = index
        while fin + 1 < len(comb_arr): 
            if comb_arr['freq'][fin + 1] == comb_arr['freq'][index]:
                fin += 1
            else:
                break            
        car2.append((comb_arr['freq'][index], sum(comb_arr['trans'][index:fin+1]) / (fin - index + 1.0)))
        index = fin + 1
        
    comb_arr = np.array(car2, dtype = [('freq', float), ('trans', float)])
    comb_arr = np.sort(comb_arr, order = 'freq')
    
    if average_data:
        values = avg_over_data([comb_arr['trans']], avgsize)[0]
        frequencies = comb_arr['freq'][avgsize:-avgsize]
    else:
        values = comb_arr['trans']
        frequencies = comb_arr['freq']
   
    if not point_start:
        point_start = 0
    if not point_end:
        point_end = len(values)
    
    plt.plot(frequencies[point_start:point_end], np.array(values[point_start:point_end]) + y_offset)

###############################################################################
    
def analyze_abfsp_data(data_sets = None, savefigs = False,
                       path = "", figsize = (10, 5), imname = "test", dpi = 300,
                       export_data = False, export_dest = "../../data/processed/",
                       title = "", plot_data = False, indices = [],
                       pad_data = False, factor = 5):
    if not data_sets:
        raise RuntimeError("Please provide data sets!")
  
    frequencies = None
    avg = []
    
    for dataset in data_sets:
        t_data, t_freqs, t_avg = load_data(path + dataset[0], title = title)
        
        ###
        t_data = t_data[dataset[2] : dataset[3]]
        t_freqs = t_freqs[dataset[2] : dataset[3]]
        t_avg = t_avg[dataset[2] : dataset[3]]
        ###
        
        if pad_data:
            t_freqs = pad_data(t_freqs, factor)
            t_avg = pad_data(t_avg, factor)
        
        if not frequencies:
            frequencies = t_freqs
        elif (len(frequencies) != len(t_freqs)) or (frequencies[0] != t_freqs[0]) or (frequencies[-1] != t_freqs[-1]):
            raise RuntimeError("Invalid data set: {}".format(dataset))
        
                
        if plot_data:
            plt.clf()
            for index in indices:
                plt.plot(t_data[index])
            plt.show()
            plt.clf()
        
        avg.append(t_avg)
        
        if export_data:
            np.array(frequencies).dump(export_dest + dataset[0] + "-f.pkl")
            np.array(t_avg).dump(export_dest + dataset[0] + "-v.pkl")
    
    values = np.zeros(len(frequencies))
        
    for index in range(len(data_sets)):
        values = values + np.array(avg[index]) * data_sets[index][1]
    
    values = values / (len(data_sets) * 1.0)
    
    plt.clf()
    plt.figure(figsize = figsize)
    
    plt.plot(frequencies, values)    
    
    if savefigs:
        plt.savefig("../../data/images/" + imname + ".png", dpi = dpi)
        
    plt.show()
    
###############################################################################

def analyze_noise(data_sets, path = "../../data/processed/noise-test/", figsize = (10, 5), 
                  imname = "unknown", point_start = None, point_end = None, savefigs = False, dpi = 300):
    
    frequencies = []
    ivalues = []
    
    for name in data_sets:
         with open(path + name + "-f.pkl", 'rb') as f:
            frequencies = pickle.load(f)
         with open(path + name + "-v.pkl", 'rb') as f:
            ivalues.append(pickle.load(f))
            
         plt.plot(frequencies, ivalues[-1])
    
    if savefigs:
        plt.savefig("../../data/images/" + imname + "-all.png", dpi = dpi)
    
    plt.show()
    
    means = []
    stds= []
    
    for freq in range(len(frequencies)):
        mean = 0.0
        for index in range(len(data_sets)):
              mean += ivalues[index][freq]
        mean = mean / len(data_sets)
        
        std = 0.0
        for index in range(len(data_sets)):
              std += (ivalues[index][freq] - mean) * (ivalues[index][freq] - mean)
        
        std = math.sqrt(std / len(data_sets))
        
        means.append(mean)
        stds.append(std)
        
        print("Frequency {}: mean {} dBV, standard deviation {}".format(frequencies[freq], mean, std))
    
    plt.clf()
    plt.errorbar(frequencies, means, yerr = stds)
    
    if savefigs:
        plt.savefig("../../data/images/" + imname + "-combined.png", dpi = dpi)
    
    plt.show()
###################################################################################################

def fourier_transform(real_data, kys, kxs): 
    
    """
    Elements of real_data are assumed to look like:
        0: X in mm
        1: Y in mm
        2: Amplitude
        3: Phase in degrees
    """
    results = []
    
    for i in range(len(kys)):
        amplitude = np.abs(np.array(
            [point[2] * np.exp(1j * (kxs[i] * (point[0] / 1000) + kys[i] * (point[1] / 1000) + (point[3] / 57.2957795131))) 
                for point in real_data]).sum())
        results.append(amplitude)
    return results

def display_chirp_scan(filename, calfilename, size, figsize, kxs, kys, reduce = 1.0, lfbeg = 0, lfend = 500):
    
    imgsize =  (int(size[0] / reduce), int(size[1] / reduce))
    
    amp_data = np.zeros(imgsize)
    phase_data = np.zeros(imgsize)
    
    with open(filename, 'rb') as f:
        scan_data = pickle.load(f)
    
    for point in scan_data:
        x = int(point[0] / reduce)
        y = int(point[1] / reduce)
        
        amp_data[x][y] = point[2]
        phase_data[x][y] = point[3]
        
    camp_data = np.zeros(imgsize)
    cphase_data = np.zeros(imgsize)
    
    with open(calfilename, 'rb') as f:
        cscan_data = pickle.load(f)
    
    for point in cscan_data:
        x = int(point[0] / reduce)
        y = int(point[1] / reduce)
        
        camp_data[x][y] = point[2]
        cphase_data[x][y] = point[3]
        
    
    
    
    plt.figure(figsize = figsize)
    plt.imshow(amp_data, cmap = 'viridis')
    plt.show()
    plt.clf()
    
    plt.figure(figsize = figsize)
    plt.imshow(phase_data, cmap = 'hsv')
    plt.colorbar()
    plt.show()
    plt.clf()
    
    ft = fourier_transform(scan_data, kys, kx = 0.0)
    plt.ylim(0.0, 27.5)
    plt.plot(ft)
    plt.ylabel('Amplitude')
    plt.xlabel('k (1/m)')
    plt.title('Fourier transform of mode at 6732 Hz')
    plt.savefig("../../data/ft-6732.png", dpi = 250)
    
    """
    plt.figure(figsize = figsize)
    plt.imshow(camp_data, cmap = 'viridis')
    plt.show()
    plt.clf()
    
    plt.figure(figsize = figsize)
    plt.imshow(cphase_data, cmap = 'hsv')
    plt.colorbar()
    plt.show()
    plt.clf()
    
    ft2d = []
    cft2d = []
    
    for kx in kxs:
        ft2d.append(fourier_transform(scan_data, kys, kx))
        cft2d.append(fourier_transform(cscan_data, kys, kx))
    """
    
    """
    plt.plot(kvecs[lfbeg:lfend], ft[lfbeg:lfend])
    plt.show()
    plt.clf()
    """
    
    """
    popt, pcov = optimize.curve_fit(_hlorentz, kvecs[lfbeg:lfend], ft[lfbeg:lfend], p0 = [17, 130, 15, 5])
    print(popt)    
     
    plt.plot([_hlorentz(k, popt[0], popt[1], popt[2], popt[3]) for k in kvecs], 
            linestyle = '--', color = 'r')
    
    plt.plot(np.array(ft) - np.array([_hlorentz(k, popt[0], popt[1], popt[2], popt[3]) for k in kvecs]), 
            linestyle = '--', color = 'g')
    """
    
    """
    plt.imshow(np.array(ft2d))
    plt.colorbar()
    plt.show()
    plt.clf()
    
    plt.imshow(np.array(cft2d))
    plt.colorbar()
    plt.show()
    plt.clf()
    
    
    plt.imshow(np.array(ft2d) - np.array(cft2d))
    plt.colorbar()
    plt.show()
    plt.clf()
    """
    
def normalize_intensity(freqRespFile, datafile, savefile, freqs):
    
    with open(freqRespFile, 'rb') as freqRes:
        freqResponseData = pickle.load(freqRes)
        frequency, response = freqResponseData
        
        for setno in range(len(freqs)):
            for index, freq in tqdm.tqdm(list(enumerate(freqs[setno]))):
                with open(datafile + '_' + str(freq) + '_' + str(setno) + '.pkl', 'rb') as f:
                    scan_data = pickle.load(f)
                    
                    norm_data = []
                    
                    for datum in scan_data:
                        datum[2] = datum[2] / response[index]
                        norm_data.append(datum)
                        
                    np.array(norm_data).dump(savefile + '_' + str(freq) + '_' + str(setno) + '.pkl')                               
                
    
def fourier_chirps(size, freqs, kxs, kys, filename, sfilename,
                   figsize = (10, 10), tag = '', dpi = 200):
    datasets = []
    args = []
    allfreqs = []

    print('Beginning chirp processing...')
    
    plt.figure(figsize = figsize)
    
    for setno in range(len(freqs)):
        for index, freq in tqdm.tqdm(list(enumerate(freqs[setno]))):
            with open(filename + '_' + str(freq) + '_' + str(setno) + '.pkl', 'rb') as f:
                scan_data = pickle.load(f)
        
            #kvecs = np.linspace(kmin, kmax, ksamples + 1)
            result = fourier_transform(real_data = scan_data, kys = kys, kxs = kxs)
        
            np.array(result).dump(sfilename + '_qsp_' + str(freq) + '_' + str(setno) + '.pkl')
            
            b = (np.arange(len(kxs)) + 1)/len(kxs)*(1.5+np.sqrt(3)/2)
            
            datasets.append(result)    
            plt.plot(b, result)
            allfreqs.append(freq)
            #args.append(np.argmax(result)  * (kmax - kmin)/ ksamples + kmin)
    
    freqmin = np.array([ np.array(row).min() for row in freqs]).min()
    freqmax = np.array([ np.array(row).max() for row in freqs]).max()
    
    plt.savefig(sfilename + "figures/chft_qspace" + tag + ".png", dpi = dpi)
    plt.show()
    plt.clf()

    plt.figure(figsize = figsize)
    plt.imshow(-1.0 * np.array(datasets), cmap = 'gray',
           origin = 'lower', extent = [b[0], b[-1], freqmin, freqmax], 
           rasterized = False, aspect = 'auto')
    plt.savefig(sfilename + "figures/chft_band" + tag + ".png", dpi = dpi)
    plt.show()
    plt.clf()
    
    plt.figure(figsize = figsize)
    plt.imshow(-1.0 * np.array(np.log(datasets)), cmap = 'gray',
           origin = 'lower', extent = [b[0], b[-1], freqmin, freqmax], 
           rasterized = False, aspect = 'auto')
    plt.savefig(sfilename + "figures/chft_band_log" + tag + ".png", dpi = dpi)
    plt.show()
    plt.clf()
    
    '''
    popt, pcov = optimize.curve_fit(fit_line, args, allfreqs)
    plt.figure(figsize = figsize)
    plt.scatter(args, allfreqs)
    #plt.plot(args, [x * popt[0] + popt[1] for x in args], 'r-')
    plt.xlabel('GKMG sweep parameter')
    plt.ylabel('frequency (Hz)')
    plt.savefig(sfilename + "figures/chft_band_auto" + tag + ".png", dpi = dpi)
    plt.show()
    
    #print("Estimated speed of air: " + str(popt[0]*2*np.pi) + " m/s")
    '''
    
def fit_line(x, k, b):
    
    return k*x+b


def _1Lorentzian(x, amplitude, mean, width):
    return amplitude * width ** 2 / ((x - mean) ** 2 + width ** 2) 

def _5Lorentzian(x, a1, m1, w1, a2, m2, w2, a3, m3, w3, a4, m4, w4, a5, m5, w5):
    return a1 * w1 ** 2 / ((x - m1) ** 2 + w1 ** 2) +\
            a2 * w2 ** 2 / ((x - m2) ** 2 + w2 ** 2) +\
            a3 * w3 ** 2 / ((x - m3) ** 2 + w3 ** 2) +\
            a4 * w4 ** 2 / ((x - m4) ** 2 + w4 ** 2) +\
            a5 * w5 ** 2 / ((x - m5) ** 2 + w5 ** 2)

def _flatfit(x, height):
    return height

def _hlorentz(x, amplitude, mean, width, height):
     return amplitude * width ** 2 / ((x - mean) ** 2 + width ** 2) + height

def flatness(kvecs, data, p0, pmin, pmax):
    
    """
    data_mean = data.mean()
    
    flatfit, flatcov = scipy.optimize.curve_fit(_flatfit, kvecs, data, p0 = [kvecs.mean()])
    dispfit, dispcov = scipy.optimize.curve_fit(_1Lorentzian, kvecs, data, p0 = [data.max(), expk, 20.0])#,
                            #bounds = ([2, expk - 30, 5], [150, expk + 30, 100]))
    
    flatvar = np.array([((data[index] - _flatfit(kvecs[index], flatfit[0])) / data_mean) ** 2 
                      for index in range(len(data))]).mean()
    
    dispvar = np.array([((data[index] - _1Lorentzian(kvecs[index], dispfit[0], dispfit[1], dispfit[2])) / data_mean) ** 2 
                      for index in range(len(data))]).mean()
    
    return dispvar / flatvar
    """
    
    
    """
    
    
    try:
        parameters, covariance = optimize.curve_fit(_hlorentz, kvecs, data, p0 = p0)
        
        
        plt.clf()    
        plt.scatter(kvecs, data) 
        
        
        data = np.array(data) - np.array([_hlorentz(k, parameters[0], parameters[1], parameters[2], parameters[3]) 
                                for k in kvecs])
    
    
          
        plt.plot(kvecs, [_hlorentz(k, parameters[0], parameters[1], parameters[2], parameters[3]) for k in kvecs], 
            linestyle = '--', color = 'r') 
        plt.plot(kvecs, data, 
            linestyle = '--', color = 'g')    
        plt.show()
        
        
        flatness = 1.0 / np.array([ (value / parameters[3]) ** 2 for value in data[pmin : pmax]]).mean()
        
        if flatness > 1000000000:
            print("Very high flatness at {} Hz".format(p0[1] * 54.11))
        
        return flatness
    except:
        print("Could not fit at frequency {}".format(p0[1] * 54.11))
        
    return -5
    """
    mean = np.array(data[pmin : pmax]).mean()
    
    return mean / math.sqrt(np.array([ ((value - mean) / mean) ** 2 for value in data[pmin : pmax]]).mean())
    
    #return 1.0 / np.array([ ((value - mean) / mean) ** 2 for value in data[pmin : pmax]]).mean()
       
def ldisp(arg):
    return 54.11 * arg  #53.28 * arg + 300.1 

def sort_arrays(order, aux):
    comb_arr = []
    
    for index in range(len(order)):
        comb_arr.append((order[index], aux[index])) 
    
    comb_arr = np.array(comb_arr, dtype = [('order', float), ('aux', float)])
    comb_arr = np.sort(comb_arr, order = 'order')
    
    rorder = []
    raux = []
    
    for index in range(len(comb_arr)):
        rorder.append(comb_arr['order'][index])
        raux.append(comb_arr['aux'][index])
        
    return np.array(rorder), np.array(raux)

def combine_data(size, freqs, 
                 #cfreqs, 
                 kmin, kmax, kmindisp, kmaxdisp,
                 ksamples, filename, figsize = (10, 10), tag = '', dpi = 200,
                 cutoff = 1000.0, rec_cutoff = 81, abs_cutoff = 2.0, meanp_cutoff = 0.5,
                 allplots = False, nimg = False, limg = False, scatter = True,
                 flatness_plot = True, normalize = True, reconstruct = False,
                 alplc = 0, alphc = 20000, flatness_peak_fit = False, flatpeak_p0 = [],
                 flat_peak_start = 0, flat_peak_end = 10000):
    datasets = []
    cdatasets = []
    
    
    args = []
    
    allfreqs = []
    callfreqs = []
    
    flats = []
    cflats = []
    
    recfr = []
    reck = []

    print('Beginning chirp processing...')
    
    plt.figure(figsize = figsize)
    
    pmin = int((kmindisp - kmin)/(kmax - kmin) * ksamples)
    pmax = int((kmaxdisp - kmin)/(kmax - kmin) * ksamples) + 1
    
    kvecs = np.linspace(kmindisp, kmaxdisp, pmax - pmin)    
    
    for setno, values in freqs:
        for freq in values:
            with open(filename + '_qsp_' + str(freq) + '_' + str(setno) + '.pkl', 'rb') as f:
                result = pickle.load(f)       
        
            if flatness:
                flats.append(flatness(kvecs = np.linspace(kmin, kmax, ksamples + 1), 
                                  data = result, 
                                  p0 = [20, freq / 54.11, 15, 5],
                                  pmin = pmin,
                                  pmax = pmax ))
        
            result = np.array(result[pmin : pmax])
            
            if normalize: result /= result.mean()
        
            datasets.append(result)    
            if allplots: 
                if (freq > alplc) and (freq < alphc):
                    plt.plot(kvecs, result)
        
            argument = np.argmax(result)  * (kmax - kmin)/ ksamples + kmin
        
            #argument -= cutoff * int(argument / cutoff)        
        
            allfreqs.append(freq)
            args.append(argument)
            
            if reconstruct:
                resmean = result.mean()
                for kp in range(int((rec_cutoff - kmin) * ksamples / (kmax - kmin))):
                    if result[kp] > abs_cutoff:
                        if result[kp] > meanp_cutoff * resmean:
                            recfr.append(freq)
                            reck.append(kp * (kmax - kmin)/ ksamples + kmin)
    """                    
    for setno, values in cfreqs:
        for freq in values:
            with open(filename + '_qsp_' + str(freq) + '_' + str(setno) + '.pkl', 'rb') as f:
                result = pickle.load(f)       
        
            if flatness:
                cflats.append(flatness(kvecs = np.linspace(kmin, kmax, ksamples + 1), 
                                  data = result, 
                                  p0 = [20, freq / 54.11, 15, 5],
                                  pmin = pmin,
                                  pmax = pmax ))
        
            result = np.array(result[pmin : pmax])
            
            if normalize: result /= result.mean()
        
            cdatasets.append(result)            
            callfreqs.append(freq)
            
    """
    freqmin = np.array(allfreqs).min()
    freqmax = np.array(allfreqs).max()
    
    if allplots:
        plt.title("Fourier transform of scan at 14071 Hz")
        plt.xlabel('ky')
        plt.ylabel('Amplitude')
        
        
        plt.savefig("../../data/chft_qspace" + tag + ".png", dpi = dpi)
        plt.show()
        plt.clf()

        plt.figure(figsize = figsize)
    
    if nimg:
        
        image = np.zeros((int(freqmax - freqmin) + 1, int(pmax - pmin) + 1))
        for index in range(len(datasets)):
            print(int(allfreqs[index] - freqmin))
            for kpoint in range(len(datasets[index])):
                image[int(allfreqs[index] - freqmin)][kpoint] += datasets[index][kpoint]
        
        
        plt.imshow(-1.0 * np.array(image), cmap = 'inferno',
                   origin = 'lower', extent = [kmin, kmax, freqmin, freqmax], 
                   rasterized = False, aspect = 'auto')
        plt.colorbar()
        
        plt.savefig("../../data/chft_band" + tag + ".png", dpi = dpi)
        plt.show()
        plt.clf()
    
        plt.figure(figsize = figsize)
    
    if limg:
        plt.imshow(-1.0 * np.array(np.log(datasets)), cmap = 'gray',
                   origin = 'lower', extent = [kmin, kmax, freqmin, freqmax], 
                   rasterized = False, aspect = 'auto')
        plt.savefig("../../data/chft_band_log" + tag + ".png", dpi = dpi)
        plt.show()
        plt.clf()
    
        plt.figure(figsize = figsize)
    
    if scatter:
        extkag_15kHz_0624
        dispx = []
        dispy = []
        
        for arg in np.unique(args):
            if (ldisp(arg) >= freqmin) and (ldisp(arg) <= freqmax):
                dispx.append(arg)
                dispy.append(ldisp(arg))
        
        plt.plot(dispx, dispy, '-r')
        plt.scatter(args, allfreqs, 
                    marker = 'o', c = 'darkslateblue', s = 5)
        
        plt.title('Frequency vs. location of ky peaks')
        plt.xlabel('ky (1/m)')
        plt.ylabel('Frequency (Hz)')
        
        
        plt.savefig("../../data/chft_band_auto" + tag + ".png", dpi = dpi)
        plt.show()
        plt.clf()
        
        plt.figure(figsize = figsize)
        
    if flatness_plot:   
        
        allfreqs, flats = sort_arrays(allfreqs, flats)
        #callfreqs, cflats = sort_arrays(callfreqs, cflats)
        
        allfreqs = allfreqs[flat_peak_start:flat_peak_end]
        flats = flats[flat_peak_start:flat_peak_end]
        
        callfreqs = np.append(callfreqs, 50000)
        """
        calibrated_flats = []
        
        cindex = 0        
        
        for index in range(len(allfreqs)):            
            while callfreqs[cindex + 1] <= allfreqs[index]:
                cindex +=1
            calibrated_flats.append(flats[index] - cflats[cindex]) 
            
        plt.scatter(allfreqs, flats - calibrated_flats, 
                               marker = 'o', c = 'darkslateblue', s = 10)
        plt.show()
        plt.clf()        
        plt.figure(figsize = figsize)
        """
        plt.ylim(0, 17)
        plt.scatter(allfreqs, flats, 
                               marker = 'o', c = 'darkslateblue', s = 10)
        
        if flatness_peak_fit:
            
            
            popt, pcov = optimize.curve_fit(_1Lorentzian, allfreqs, flats, p0 = flatpeak_p0)
            print(popt)         
            
            """
            popt = [15.89960297, 13954.61352051, 42.18558273]
            """
            
            plt.plot(allfreqs, [_1Lorentzian(f, popt[0], popt[1], popt[2]) for f in allfreqs], 
                           linestyle = '--', color = 'r')
        
        
        
        """
        #FIND FLAT BANDS
        for index in range(len(allfreqs)):
            if allfreqs[index] > 6720.0:
                if allfreqs[index] < 6740.0:
                    print(allfreqs[index], flatness(kvecs, datasets[index], allfreqs[index] / 54.11))
        """
        
        
        plt.title('Flatness score vs. frequency')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Score (arbitrary units)')
        
        plt.savefig("../../data/chft_flatness" + tag + ".png", dpi = dpi)
        plt.show()
        plt.clf()
        
        plt.figure(figsize = figsize)
        
    if reconstruct:
        plt.scatter(reck, recfr, marker = 'o', c = 'darkslateblue', s = 1)
        
        plt.savefig("../../data/chft_reconstruct" + tag + ".png", dpi = dpi)
        plt.show()
        plt.clf()
        
        plt.figure(figsize = figsize)

def _2Lorentzian(x, amplitude1, mean1, width1, amplitude2, mean2, width2):
    return amplitude1 * width1 ** 2 / ((x - mean1) ** 2 + width1 ** 2) +\
                amplitude2 * width2 ** 2 / ((x - mean2) ** 2 + width2 ** 2)
