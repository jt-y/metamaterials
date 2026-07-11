#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:58:07 2019

@author: radu
"""

#import stmpy
import os
import pickle
import glob
import skimage
import numpy as np
import math
import imageio
from matplotlib import pyplot as plt
from matplotlib import patches as ptc
from matplotlib import colors as colors

slst6_lattice = [(( 0.0000,  0.0000), 0.4),
                 (( 0.0000,  0.7500), 0.1),
                 ((-0.2165,  0.8750), 0.1),
                 ((-0.4330,  0.7500), 0.1),
                 ((-0.6495,  0.6250), 0.1),
                 ((-0.6495,  0.3750), 0.1),
                 ((-0.8660,  0.2500), 0.1),
                 ((-0.8660,  0.0000), 0.1),
                 ((-0.8660, -0.2500), 0.1),
                 ((-0.6495, -0.6250), 0.1),
                 ((-0.6495, -0.3750), 0.1),
                 ((-0.2165, -0.8750), 0.1),
                 ((-0.4330, -0.7500), 0.1),
                 (( 0.0000, -0.7500), 0.1),
                 (( 0.2165,  0.8750), 0.1),
                 (( 0.4330,  0.7500), 0.1),
                 (( 0.6495,  0.6250), 0.1),
                 (( 0.6495,  0.3750), 0.1),
                 (( 0.8660,  0.2500), 0.1),
                 (( 0.8660,  0.0000), 0.1),
                 (( 0.8660, -0.2500), 0.1),
                 (( 0.6495, -0.6250), 0.1),
                 (( 0.6495, -0.3750), 0.1),
                 (( 0.2165, -0.8750), 0.1),
                 (( 0.4330, -0.7500), 0.1)]

def compile_data_to_array(data_dir, sample_start=5500, sample_end=5700):
    fnames = list(sorted(glob.glob(os.path.join(data_dir, "*.pkl"))))
    
    print('Found %s records' % len(fnames))
    # Load into a list of tuples of xmin, xmax, y, data
    data = []
    XMIN = None
    XMAX = None
    for fname in fnames:
        with open(fname, 'rb') as f:
            fft_data = pickle.load(f)
          
        # Isolate the frequency. In our case, 28kHz is usually around sample 8000 to 10000
        amplitudes = []
        for i in range(len(fft_data)):
            intar = np.array(fft_data[i][sample_start:sample_end])
            amplitudes.append(intar.max())               
        
        name = os.path.basename(fname).replace('.pkl', '').replace('continuous_', '')
        coords = [float(coord) for coord in name.split('_')]
        xmin, xmax, y = coords
        XMIN = xmin
        XMAX = xmax
        data.append((xmin, xmax, y, amplitudes))
        
    # Sort by y coordinate (xmin and xmax are expected to be the same for all)
    data = list(sorted(data))
    if not data:
        raise RuntimeError('No Data Found')
       
    
    
    # Just get the amplitudes and stack them on each other to form an image
    ampdata = [d[-1] for d in data]

    # Get the minimum size of any of these, so we can interpolate to a fixed array length
    target_size = int(np.median(np.array([len(x) for x in ampdata])))
    print('Median number of records in continguous strip: %s' % str(target_size))
    resized_ampdata = [np.interp(np.linspace(XMIN, XMAX, target_size),
                                 np.linspace(XMIN, XMAX, len(d)), d)
                       for d in ampdata]
    resized_ampdata = np.array(resized_ampdata)

    return resized_ampdata

def load_lockin_data(data_dir, cutoff = 0):
    fnames = list(sorted(glob.glob(os.path.join(data_dir, "*.pkl"))))
    
    print('Found %s records' % len(fnames))
    # Load into a list of tuples of xmin, xmax, y, data
    data = []
    
    for fname in fnames:
        with open(fname, 'rb') as f:
            lockin_data = pickle.load(f)
          
        # Separate the amplitude and phase data
        amplitudes = []
        phases = []
        
        for point in lockin_data:
            amplitudes.append(float(point[0]))
            phases.append(float(point[1]))               
        
        name = os.path.basename(fname).replace('.pkl', '').replace('continuous_', '')
        coords = [float(coord) for coord in name.split('_')]
        xmin, xmax, y = coords
        
        amplitudes = amplitudes[cutoff:]
        phases = phases[cutoff:]
        
        if xmax < 0:
            xmax *= -1
            amplitudes = amplitudes[::-1]
            phases = phases[::-1]

        data.append((xmin, xmax, y, amplitudes, phases))
        
    # Sort by y coordinate (xmin and xmax are expected to be the same for all)
    data = list(sorted(data))
    if not data:
        raise RuntimeError('No Data Found')   
    
    # Get the amplitudes and phases and stack them on each other to form an image
    phase_data = [d[-1] for d in data]
    amp_data = [d[-2] for d in data]
    
    # Get the minimum size of any of these, so we can interpolate to a fixed array length
    target_size = np.array([len(x) for x in amp_data]).min()
    print('Minimum number of records in continguous strip: %s' % str(target_size))
    ampdata = np.array([d[:target_size] for d in amp_data])
    phasedata = np.array([d[:target_size] for d in phase_data])

    return ampdata, phasedata
    
def eliminate_null_measurements(data):
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            if data[i, j] == -1:
                neighbors = [(i+di, j+dj) for di in [-1, 1] for dj in [-1, 1]]
                neighbors = list(filter(lambda x: 0 <= x[0] < h, neighbors))
                neighbors = list(filter(lambda x: 0 <= x[1] < w, neighbors))
                # Don't average over bad results nearby either
                neighbors = list(filter(lambda x: data[x] != -1, neighbors))
                # print(neighbors)
                data[i, j] = sum([data[n] for n in neighbors]) / len(neighbors)
    return data

def fit_plane(data):
    s1 = 0.0
    sx = 0.0
    sy = 0.0
    sz = 0.0
    sxx = 0.0
    sxy = 0.0
    sxz = 0.0
    syy = 0.0
    syz = 0.0
    
    for x in range(len(data)):
        for y in range(len(data[x])):
            s1 += 1.0
            sx += x
            sy += y
            sz += data[x][y]
            sxx += x * x
            sxy += x * y
            sxz += x * data[x][y]
            syy += y * y
            syz += y * data[x][y]
            
    coeff = np.array([[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, s1]])
    fterm = np.array([sxz, syz, sz])
    
    solution = np.linalg.solve(coeff, fterm)
    
    if np.allclose(np.dot(coeff, solution), fterm):
        print("Successfully fitted plane with a={}, b={}, c={}".format(solution[0], 
              solution[1], solution[2]))
    else:
        print("Error while fitting plane!")
    
    return solution

def subtract_plane(data):
    sol = fit_plane(data)
    for x in range(len(data)):
        for y in range(len(data[x])):
            data[x][y] -= (sol[0] * x + sol[1] * y + sol[2])
    return data

def compile_direct_scans(dataset, names, tags, filenames, sub_plane = False, crop = False,
                         cr_x_st = 0, cr_x_end = 1, cr_y_st = 0, cr_y_end = 1,
                         data_min_offset = 0.0, data_max_offset = 0.0, sample_start = 5,
                         sample_end = 195, size = (10, 10), resize = False, 
                         cmap = 'viridis', savefigs = False,
                         xlabel = "x-distance (mm)", ylabel = "y-distance (mm)", dpi = 300,
                         plot_lattice = False, lattice_type = slst6_lattice, scale = 1.0,
                         lxof = 0.0, lyof = 0.0, unit_cell = 15.0, angle = 0.0, reduce = 4):
    
    figsize = (size[1] / reduce, size[0] / reduce)
    newsize = (size[0] * scale, size[1] * scale)
    
    for index in range(len(names)):
        data = compile_data_to_array("../data/" + dataset + "-" + names[index],
                                     sample_start = sample_start, sample_end = sample_end)
        
        if crop:
            newdata = []
            xst = int(cr_x_st * len(data))
            xend = int(cr_x_end * len(data))
            yst = int(cr_y_st * len(data[0]))
            yend = int(cr_y_end * len(data[0]))
            for i in range(xst, xend):
                newdata.append(np.array(data[i][yst : yend]))
            data = np.array(newdata)
            
        if sub_plane:
            data = subtract_plane(data)

        plt.figure(figsize = figsize)

        if resize:
            plt.imshow(skimage.transform.resize(data, newsize), cmap = cmap, 
                   vmin = data.min() + data_min_offset, vmax = data.max() + data_max_offset)
        else:
            plt.imshow(data, cmap = cmap, 
                   vmin = data.min() + data_min_offset, vmax = data.max() + data_max_offset)
        
        ### OVERLAP IMAGE OF LATTICE        
        if plot_lattice:
            fig = plt.gcf()
            ax = fig.gca()
        
            for coord in lattice_type:
                for xi in range(int(size[0] / unit_cell) + 1):
                    for yi in range(int(size[1] / unit_cell) + 1):
                        
                        xl = (coord[0][0] + xi * math.sqrt(3) + lxof) * unit_cell * scale
                        yl = (coord[0][1] + yi * 3 + lyof) * unit_cell * scale
                        
                        xr = xl * math.cos(angle) - yl * math.sin(angle) 
                        yr = xl * math.sin(angle) + yl * math.cos(angle) 
                        
                        circle = plt.Circle((xr, yr), coord[1] * unit_cell * scale, edgecolor='black', 
                                    fill = False, alpha = 0.5)
                        ax.add_artist(circle)
                        
                        ###
                        
                        xl = (coord[0][0] + (xi - 0.5) * math.sqrt(3) + lxof) * unit_cell * scale
                        yl = (coord[0][1] + (yi + 0.5) * 3 + lyof) * unit_cell * scale
                        
                        xr = xl * math.cos(angle) - yl * math.sin(angle) 
                        yr = xl * math.sin(angle) + yl * math.cos(angle)
                        
                        circle = plt.Circle((xr, yr), coord[1] * unit_cell * scale, edgecolor='black', 
                                            fill = False, alpha = 0.5)
                        ax.add_artist(circle)
                
        plt.title(tags[index])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.colorbar()
        
        if savefigs:
            plt.savefig("../data/" + filenames[index] + ".png", dpi = dpi)
            
        plt.show()
        plt.clf()

def crop_data(data, crop_size):
    raise NotImplementedError
    
    """
    newdata = []
            xst = int(cr_x_st * len(data))
            xend = int(cr_x_end * len(data))
            yst = int(cr_y_st * len(data[0]))
            yend = int(cr_y_end * len(data[0]))
            for i in range(xst, xend):
                newdata.append(np.array(data[i][yst : yend]))
            data = np.array(newdata)
    """
  
def draw_lattice(lattice_type, size, unit_cell, lattice_offset, scale, angle, aspectx, aspecty):
    fig = plt.gcf()
    ax = fig.gca()
        
    for coord in lattice_type:
        for xi in range(int(size[0] / unit_cell) + 1):
            for yi in range(int(size[1] / unit_cell) + 1):
                        
                xl = (coord[0][0] + xi * math.sqrt(3) + lattice_offset[0]) * unit_cell * scale
                yl = (coord[0][1] + yi * 3 + lattice_offset[1]) * unit_cell * scale
                        
                xr = xl * math.cos(angle) - yl * math.sin(angle) 
                yr = xl * math.sin(angle) + yl * math.cos(angle)
                
                xr = xr / aspectx
                yr = yr / aspecty
                        
                circle = ptc.Ellipse((xr, yr), 
                                     width = 2 * coord[1] * unit_cell * scale / aspectx, 
                                     height = 2 * coord[1] * unit_cell * scale / aspecty, 
                                     edgecolor='black', 
                                    fill = False, alpha = 0.5)
                ax.add_artist(circle)
                        
                        ###
                        
                xl = (coord[0][0] + (xi - 0.5) * math.sqrt(3) + lattice_offset[0]) * unit_cell * scale
                yl = (coord[0][1] + (yi + 0.5) * 3 + lattice_offset[1]) * unit_cell * scale
                        
                xr = xl * math.cos(angle) - yl * math.sin(angle) 
                yr = xl * math.sin(angle) + yl * math.cos(angle)
                
                xr = xr / aspectx
                yr = yr / aspecty
                        
                circle = ptc.Ellipse((xr, yr), 
                                     width = 2 * coord[1] * unit_cell * scale / aspectx, 
                                     height = 2 * coord[1] * unit_cell * scale / aspecty, 
                                     edgecolor='black',
                                     fill = False, alpha = 0.5)
                ax.add_artist(circle)
 
def volts_to_decibels(data):
    return 8.68589 * np.log(data + 0.00001)# - data.min() + 0.01)

def scta(s, c):
    factor = (s**2 + c**2)**0.5
    s /= factor
    c /= factor
    angle = np.arccos(c)
    if (s > 0): angle *= -1.0
    return angle / 0.01745329252 

def resize_phase_data(data, size):
    cosine_data = skimage.transform.resize(np.cos(data * 0.01745329252), size)
    sine_data = skimage.transform.resize(np.sin(data * 0.01745329252), size)
    
    resized_data = [[scta(cosine_data[y][x], sine_data[y][x]) for x in range(len(cosine_data[y]))] for y in range(len(cosine_data))]
    
    return resized_data

def true_phase(raw):
    while raw < -180.0:
        raw += 360.0
    while raw > 180.0:
        raw -= 360.0
    return raw
 
def animate_wave(amps, phases, cmap, vmin, vmax, frames, period, anim_dpi, anim_lattice,
                 lattice_type, size, unit_cell, lattice_offset, scale, angle,
                 anim_fname, aspect, labels, title, aspectx = 1.0, aspecty = 1.0):
    with imageio.get_writer('../data/' + anim_fname, mode='I', duration = period / frames) as writer:
        for offset in np.linspace(-180.0, 180.0, frames, endpoint = False):
            data = [[amps[x][y] * math.cos((phases[x][y] + offset) * 0.01745329252) 
                for y in range(len(amps[x]))] 
                    for x in range(len(amps))]
        
            plt.clf()
            plt.imshow(data, cmap = cmap, vmin = vmin, vmax = vmax, aspect = aspect)
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(title)
            if anim_lattice: draw_lattice(lattice_type, size, unit_cell, lattice_offset, scale, angle, aspectx, aspecty)
            plt.colorbar()
            plt.savefig('../data/animations/temp.png', dpi = anim_dpi)
            writer.append_data(imageio.imread('../data/animations/temp.png'))

def mask_data(amp_data, size, unit_cell, aspectx, aspecty, real_size, lattice_offset, lattice_type, angle):
    mask_data = generate_lattice_mask(size = size, 
                                              unit_cell = unit_cell, 
                                              aspectx = aspectx,
                                              aspecty = aspecty,
                                              real_size = real_size, 
                                              lattice_offset = lattice_offset, 
                                              lattice_type = lattice_type, 
                                              angle = angle)
    return np.multiply(amp_data, mask_data)

def compile_lockin_scans(datasets, load_cutoff = 0,
                         sub_plane = False,  mask = False,
                         crop = False, crop_size = ((0, 0), (1, 1)), 
                         log_scale = True, amp_data_offset = (0, 0),
                         size = (10, 10), resize = False, 
                         amp_cmap = 'viridis', phase_cmap = 'hsv', savefigs = False,
                         labels = ('x-distance (mm)', 'y-distance (mm)'), dpi = 300,
                         plot_lattice = False, lattice_type = slst6_lattice, lattice_offset = (0, 0),
                         scale = 1.0, unit_cell = 30.0, angle = 0.0, reduce = 4,
                         animate = False, anim_dpi = 10, anim_lattice = False,
                         anim_frames = 36, anim_period = 1.0, anim_cmap = 'viridis'):
    
    """ 
    DESCRIPTION OF PARAMETERS:
        
        @param load_cutoff: Temporary fix to sheared raw data; ideally, will be fixed from scan script
    
        @param sub_plane: (bool) Decides if method fits plane to data and subtracts it, 
                                to eliminate background
    
        @param crop: (bool) Decides whether the image will be cropped
        (TO BE DEVELOPED)
    
        @param crop_size: (tuple) Coordinates of rectangle to which images will be cropped
                        format: ((x_start, y_start), (x_end, y_end))
    
        @param data_offset: Offsets to be added to minimum and maximum of data in colormap
    
        @param size: Size of scan, in format (x, y)
    
        @param resize: Decides whether data is resized to the aspect ratio of the actual scan
    
        @param cmap: Color map used for plots
    
        @param savefigs: Decides whether to save figures
    
        @param labels: X and Y-axis labels for plots
    
        @param dpi: Resolution for saving plots
    
        @param plot_lattice: Decides whether to superimpose lattice onto image
        (TO BE DEVELOPED)
    
        @param lattice_type: Decides what type of lattice will be plotted
    
        @param scale: Scale of image relative to scan size
    
        @param lattice_offset: Offset of lattice sketch relative to origin of image
    
        @param unit_cell: Size of unit cell for lattice
    
        @param angle: Angle by which lattice is rotated
    
        @param reduce: Factor by which figure size is reduced relative to scan size
    
        @param dataset: (array) Array of info about scans to be processed. Each scan's info is a tuple:
                (
                name of dataset folder, 
                (titles to be given to plot), 
                (filenames for saved images)
                )
    
    
        TODO: log scale for amps, true resize for phases 
    """    
        
    figsize = (size[0] / reduce, size[1] / reduce)
    new_img_size = (size[1] * scale, size[0] * scale)
    
    
    for scan in datasets:
        
        #LOAD AND PROCESS DATA
        amp_data, phase_data = load_lockin_data("../data/" + scan[0], cutoff = load_cutoff) 
        
        
        aspectx = size[0] / len(amp_data[0])
        aspecty = size[1] / len(amp_data)
        aspect = aspecty / aspectx
        
        plt.figure(figsize = figsize)
        
        if crop: 
            amp_data = crop_data(amp_data, crop_size)  
            phase_data = crop_data(phase_data, crop_size)
        
        if mask: amp_data = mask_data(amp_data = amp_data, 
                                 size = size, 
                                 unit_cell = unit_cell, 
                                 aspectx = aspectx, 
                                 aspecty = aspecty, 
                                 real_size = amp_data.shape, 
                                 lattice_offset = lattice_offset, 
                                 lattice_type = lattice_type, 
                                 angle = angle)
            
        if log_scale: amp_data = volts_to_decibels(amp_data)        
        if sub_plane: amp_data = subtract_plane(amp_data)  
        
        ### PLOT AMPLITUDE DATA   
        
        
        if resize:
            plt.imshow(skimage.transform.resize(amp_data, new_img_size), cmap = amp_cmap, 
                   vmin = amp_data.min() + amp_data_offset[0], 
                   vmax = amp_data.max() + amp_data_offset[1])
        else:
            plt.imshow(amp_data, cmap = amp_cmap, 
                   vmin = amp_data.min() + amp_data_offset[0], 
                   vmax = amp_data.max() + amp_data_offset[1],
                   aspect = aspect)
        
        if plot_lattice: draw_lattice(lattice_type, size, unit_cell, lattice_offset, scale, angle, aspectx, aspecty)
        plt.title(scan[1][0])
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        plt.colorbar()
        
        if savefigs:
            plt.savefig("../data/" + scan[2][0] + ".png", dpi = dpi)
            
        plt.show()
        plt.clf()
        
        
        ### PLOT PHASE DATA   
        
        plt.figure(figsize = figsize)
        if resize:
            plt.imshow(skimage.transform.resize(phase_data, new_img_size), cmap = phase_cmap, 
                   vmin = -180.0, vmax = 180.0)
        else:
            plt.imshow(phase_data, cmap = phase_cmap, 
                   vmin = -180.0, vmax = 180.0, aspect = aspect)
        
        if plot_lattice: draw_lattice(lattice_type, size, unit_cell, lattice_offset, scale, angle, aspectx, aspecty)
        plt.title(scan[1][1])
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        plt.colorbar()
        
        if savefigs:
            plt.savefig("../data/" + scan[2][1] + ".png", dpi = dpi)
            
        plt.show()
        plt.clf()
    
        ### ANIMATE DATA
        if animate:
            amp_data = amp_data - amp_data.min()
        
            animate_wave(amps = amp_data, 
                     phases = phase_data, 
                     cmap = anim_cmap, 
                     vmin = (-1) * amp_data.max(), 
                     vmax = amp_data.max(), 
                     frames = anim_frames, 
                     period = anim_period,
                     anim_dpi = anim_dpi,
                     anim_lattice = anim_lattice,
                     
                     lattice_type = lattice_type, size = size, unit_cell = unit_cell, 
                     lattice_offset = lattice_offset, scale = scale, angle = angle, 
                     aspectx = aspectx, aspecty = aspecty,
                     
                     anim_fname = scan[4], aspect = aspect, labels = labels, title = scan[3])
        
########################################################################################################

########################################################################################################

def modified_fourier_transform(real_data, real_size, kvecs):
    sums = [row.sum() for row in real_data]
    results = []
    window = np.hanning(len(sums))
    for k in kvecs:
        amplitude = np.abs(np.array([window[yin] * sums[yin] * np.exp(1j * k * 
                                     (yin * real_size[1] / len(sums))) 
                    for yin in range(len(sums))]).sum())
        results.append(amplitude)
    return results

def generate_lattice_mask(size, unit_cell, aspectx, aspecty, real_size, lattice_offset, lattice_type, angle):
    lattice_mask = np.zeros(real_size)
    for coord in lattice_type:
        for xi in range(int(size[0] / unit_cell) + 1):
            for yi in range(int(size[1] / unit_cell) + 1):
                        
                xl = (coord[0][0] + xi * math.sqrt(3) + lattice_offset[0]) * unit_cell
                yl = (coord[0][1] + yi * 3 + lattice_offset[1]) * unit_cell
                        
                xr = xl * math.cos(angle) - yl * math.sin(angle) 
                yr = xl * math.sin(angle) + yl * math.cos(angle)
                
                xr = xr / aspectx
                yr = yr / aspecty
                      
                xr = int(xr)
                yr = int(yr)

                
                for x in range(xr - 1, xr + 3):
                    for y in range(yr - 1, yr+ 3):
                
                        if (x >= 0) and (y >= 0):
                            try:
                        
                                lattice_mask[y][x] = 1
                            except:
                                a = 0
                        ###
                        
                xl = (coord[0][0] + (xi - 0.5) * math.sqrt(3) + lattice_offset[0]) * unit_cell 
                yl = (coord[0][1] + (yi + 0.5) * 3 + lattice_offset[1]) * unit_cell 
                        
                xr = xl * math.cos(angle) - yl * math.sin(angle) 
                yr = xl * math.sin(angle) + yl * math.cos(angle)
                
                xr = xr / aspectx
                yr = yr / aspecty
                     
                xr = int(xr)
                yr = int(yr)
                
                for x in range(xr - 1, xr + 3):
                    for y in range(yr - 1, yr+ 3):
                
                        if (x >= 0) and (y >= 0):
                            try:
                        
                                lattice_mask[y][x] = 1
                            except:
                                a = 0
    
    return lattice_mask

def reciprocal_space_info(real_data, unit_cell = 30, real_size = (200, 200)):
    

    #real_data = np.resize(real_data, (3000, 112))
    reciprocal_data = np.abs(np.fft.fft2(real_data))
    
    
    #Looking only at the maximum right now
    result = np.argwhere(reciprocal_data.max() == reciprocal_data)[0][0]
    if result > len(real_data) / 2:
        result -= len(real_data)
    return result

def band_structure(datasets, load_cutoff = 5):
    """
    dataset contains:
        0: frequency
        1: filename
    """
    kvecs = []
    freqs = []
    
    for scan in datasets:
        amp_data, phase_data = load_lockin_data("../data/" + scan[1], cutoff = load_cutoff)     
        real_data = amp_data * np.exp(1j * phase_data / 57.2957795131)
        kvecs.append(reciprocal_space_info(real_data))
        freqs.append(scan[0])

    plt.clf()
    plt.scatter(kvecs, freqs)
    plt.show()


