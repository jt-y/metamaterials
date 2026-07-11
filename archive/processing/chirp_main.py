import archive.processing.chirp_functions as proc
import numpy as np

branch = 8

if branch == 0: # Noise pre-processing  
    data_sets = []
    for index in range(1):
        data_sets.append(("noise-test/noise-test-{}".format(index + 1), 1.0, 1, 21))


    proc.analyze_abfsp_data(data_sets = data_sets, savefigs = False, path = "../data/", 
                   figsize = (10, 5), imname = "l4", dpi = 300, export_data = True)

if branch == 1: # Noise analysis
    data_sets = []
    for index in range(100):
        data_sets.append("noise-test-{}".format(index + 1))

    proc.analyze_noise(data_sets, path = "../data/processed/noise-test/", figsize = (10, 5), 
                  imname = "100ns", point_start = None, point_end = None, savefigs = True)

if branch == 2: # Combine processed, difference l-d
    data_sets = [("ab-fs-l-1", 1.0), ("ab-fs-d-1", -1.0), 
                 ("ab-fs-l-2", 1.0), ("ab-fs-d-2", -1.0),
                 ("ab-fs-l-3", 1.0), ("ab-fs-d-3", -1.0)]

    proc.combine_multiple_measurements(data_sets, path = "../data/processed/", avgsize = 1,
                                  average_data = False, figsize = (10, 5), imname = "dif-1,2,3",
                                  point_start = None, point_end = None, savefigs = True)

if branch == 3: # Combine processed, add up
    data_sets = [("ab-fs-l-1", 1.0), ("ab-fs-d-1", 1.0), 
                 ("ab-fs-l-2", 1.0), ("ab-fs-d-2", 1.0),
                 ("ab-fs-l-3", 1.0), ("ab-fs-d-3", 1.0),
                 ("ab-fs-l-4", 1.0)]

    proc.combine_multiple_measurements(data_sets = [("ab-fs-large-cs-1", 1.0), ("ab-fs-large-cs-2", 1.0)], 
                                               path = "../data/processed/", avgsize = 15,
                                  average_data = False, figsize = (10, 5), imname = "cs-trans",
                                  point_start = -60, point_end = -30, savefigs = False,
                                  title = "Transmission behind large site")

if branch == 4: # Combine abfsp and regular transmittance
    
    data_sets = [("ab-fs-l-1", 1.0), ("ab-fs-d-1", 1.0), 
                 ("ab-fs-l-2", 1.0), ("ab-fs-d-2", 1.0),
                 ("ab-fs-l-3", 1.0), ("ab-fs-d-3", 1.0),
                 ("ab-fs-l-4", 1.0), ("ab-fs-l-5", 1.0)]

    proc.combine_multiple_measurements(data_sets=[], path = "../data/processed/", avgsize = 5,
                                  average_data = True, figsize = (10, 5), imname = "cmm-1-5-us",
                                  point_start = None, point_end = None, savefigs = False, 
                                  showplot = False,
                                  title = "Response behind material vs. frequency")
    
    data_sets = [("bfs-slst6-1", 1.0), ("bfs-slst6-1-cal", -1.0)]

    proc.plot_transmittance_measurements(data_sets, path = "../data/processed/transmittance/", avgsize = 25,
                                  average_data = True, point_start = None, point_end = None,
                                  y_offset = 0.0)
    
    
    #plt.savefig("../data/images/" + imname + ".png", dpi = 300)
    plt.show()

if branch == 5: # Regular pre-processing
    proc.analyze_abfsp_data(data_sets = [("ab-fs-large-cs-2", 1.0, 1, -1)], savefigs = False, path = "../data/", 
                   figsize = (10, 5), imname = "hfb", dpi = 300, export_data = True, title = "abfs_",
                   plot_data = True, indices = range(80, 86),
                   pad_data = False, factor = 8)
    
if branch == 6: # Chirp pre-processing
    proc.abfsp.preprocess_chirp(data_dir = '../data/scpfd-small-1/', 
                       frequencies = np.linspace(13700, 14400, 701), 
                       targetfile = '../data/processed/scpfd-small-freqspec/scpfd', 
                       title = 'scppos_', setno = 1)

if branch == 7: # Chirp Fourier
    #proc.display_chirp_scan(filename = '../data/processed/scpf/scpf_13000.0.pkl', size = (220, 220),
    #                   figsize = (10, 10), reduce = 2.0, kvecs = np.linspace(0, 200, 201))
    
    """
    freqs = np.array([np.linspace(5500, 15500, 2001),
                      np.linspace(6700, 7200, 251),
                      np.linspace(11000, 12000, 501),
                      np.linspace(5502.5, 15502.5, 2001)])
    """
    freqs = np.array([[], np.linspace(13700, 14400, 701)])
    
    proc.fourier_chirps(size = (220, 220), 
                   freqs = freqs, 
                   kmin = 0, 
                   kmax = 1000, 
                   ksamples = 2001, 
                   filename = '../data/processed/scpfd-small-freqspec/scpfd',
                   sfilename = '../data/processed/scpfd-small-qsp/scpfd',
                   tag = '_scpfd')

if branch == 8: # Combine measurements
    
    """
    #SCP0:
    freqs = np.array([np.linspace(6000, 13000, 701),
                      np.linspace(11000, 12000, 201),
                      np.linspace(9100.1, 9500.1, 81),
                      np.linspace(7500, 10000, 1251),
                      np.linspace(13000, 15500, 251),
                      np.linspace(6700, 7200, 201)])
    """
    
    """
    #SCPF2:
    freqs = np.array([np.linspace(5500, 15500, 2001),
                      np.linspace(6700, 7200, 251),
                      np.linspace(11000, 12000, 501),
                      np.linspace(5502.5, 15502.5, 2001)])
    """
    
    """
    #SCPFD:
    freqs = np.array([(0, np.linspace(6700, 6850, 151)), 
                      (1, np.linspace(5000, 15000, 1001)), 
                      (2, np.linspace(11000, 12000, 251)),
                      (3, np.linspace(6724, 6744, 201)),
                      (4, np.linspace(6731.8, 6732.1, 61)),
                      (5, np.linspace(6725, 6740, 301)),
                      (6, np.linspace(11448, 11468, 401)),
                      (7, np.linspace(6600, 7000, 801)),
                      (8, np.linspace(11200, 11500, 451))
                      ])
    """
    freqs = np.array([(9, np.linspace(5000, 15000, 201))])
    """ 
    
    freqs = np.array([(0, np.linspace(10000, 25000, 1501)),
                      (1, np.linspace(13700, 14400, 701))
                      ])
    """
    proc.combine_data(size = (220, 220), 
                 freqs = freqs, 
                 #cfreqs = cfreqs,
                 kmin = 0, 
                 kmax = 500, 
                 kmindisp = 0, 
                 kmaxdisp = 300,
                 ksamples = 2001, 
                 #filename = '../data/processed/scpfd-small-qsp/scpfd', 
                 filename = '../data/processed/scpfd-qsp/scpfd',
                 figsize = (10, 10), 
                 tag = '_scpfdcal', 
                 dpi = 200,
                 cutoff = 1000.0,
                 rec_cutoff = 81, abs_cutoff = 1.5, meanp_cutoff = 1.5,
                 allplots = False, 
                 alplc = 0, alphc = 24072,
                 nimg = False, 
                 limg = False, 
                 scatter = True,
                 flatness_plot = True, 
                 normalize = False,
                 reconstruct = False,
                 flatness_peak_fit = False,
                 flatpeak_p0 = [25, 12200, 50],
                 flat_peak_start = 0, flat_peak_end = 500)

if branch == 9: # Visualisation
    proc.display_chirp_scan(filename = '../data/processed/scpfd-freqspec/scpfd_6732.0_3.pkl', 
                       calfilename = '../data/processed/scpfd-freqspec/scpfd_7000.0_9.pkl',
                       size = (220, 220), 
                       figsize = (10, 10), 
                       kxs = np.linspace(0, 200, 201), 
                       kys = np.linspace(0, 275, 276),
                       reduce = 1.0,
                       lfbeg = 0,
                       lfend = -1)
    

"""
("ab-fs-l-1", 1.0, 3001, 4001), ("ab-fs-d-1", -1.0, 3001, 4001), 
             ("ab-fs-l-2", 1.0, 3001, 4001), ("ab-fs-d-2", -1.0, 3001, 4001),        
             ("ab-fs-l-3", 1.0, 1, 1001), ("ab-fs-d-3", -1.0, 1, 1001)
             
             ("ab-fs-l-1", 1.0, 1, 6001), ("ab-fs-d-1", -1.0, 1, 6001), 
             ("ab-fs-l-2", 1.0, 1, 6001), ("ab-fs-d-2", -1.0, 1, 6001)


"""

             

