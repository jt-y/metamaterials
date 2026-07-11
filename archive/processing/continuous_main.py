import archive.processing.continuous_functions as proc

"""
dataset = "ds-large"
names = []
tags = []
filenames = []
for index in range(1):
    names.append(str(index + 4))
    tags.append("Direct scan above large SLST6 lattice")
    filenames.append(dataset + "-" + names[index])

compile_direct_scans(dataset = dataset, names = names, tags = tags, filenames = filenames, 
                     sub_plane = True, crop = False, cr_x_st = 0.25, cr_x_end = 0.75, 
                     cr_y_st = 0.55, cr_y_end = 0.85, data_min_offset = 0.0, data_max_offset = 0.0, 
                     sample_start = 0, sample_end = 400, size = (90, 90), resize = True, 
                     cmap = 'viridis', savefigs = False, unit_cell = 30.0, reduce = 7.5,
                     xlabel = "X-distance (x0.05 mm)", ylabel = "Y-distance (x0.05 mm)", dpi = 300,
                     plot_lattice = True, lattice_type = slst6_lattice, scale = 10.0, angle = 0.0,
                     lxof = -0.232, lyof = -1.5)

#-0.48, -2

# -0.232, -1.5
"""

"""
amps, phases = load_lockin_data("../data/lds-test-3", cutoff = 5)

#print(amps)
plt.figure(figsize = (10, 7.5))
#plt.imshow(amps, cmap = 'viridis')
plt.imshow(skimage.transform.resize(amps, (600, 800)), cmap = 'inferno')


#print(phases)
#plt.imshow(phases, cmap = 'hsv', vmin = -180.0, vmax = 180.0)
#plt.imshow(skimage.transform.resize(phases, (600, 800)), cmap = 'hsv', vmin = -180.0, vmax = 180.0)
plt.show()


[('ldsf-test', 
                                  ('Direct scan of transducer only, amplitude',
                                   'Direct scan of transducer only, phase'), 
                                   ('ldsf-tr-amp', 'ldsf-tr-phs'),
                                   'Real space, transducer only, 8000 Hz'
                                )]
                                  
[('ldsf-disp/ldsf-5400.0', 
                                  ('Direct scan of SLST6 at 5400 Hz, amplitude',
                                   'Direct scan of SLST6 at 5400 Hz, phase'), 
                                   ('lds-5400-amp', 'lds-5400-phs'),
                                   'Real space, SLST6, 5400 Hz',
                                   'slst6-5400.gif'
                                )], 
"""



 #IMAGING DIRECT SCANS
frequencies = np.linspace(6730, 6735, 6)
frequencies = [13955.0]
datasets = []

for freq in frequencies:
    datasets.append(
            ('ldsfdet-' + str(freq), 
                     ('Direct scan of large SLST6 at {} Hz, sound intensity'.format(freq), 
                      'Direct scan of large SLST6 at {} Hz, phase'.format(freq)),
                     ('lds-amp-' + str(freq),
                      'lds-phs-' + str(freq)),
                      
                      'Real space, SLST6, {} Hz'.format(freq),
                      'slst6-anim-{}hz.gif'.format(int(freq))
            ))

proc.compile_lockin_scans(datasets = datasets, 
        load_cutoff = 5, mask = True,
        sub_plane = False, 
        crop = False, crop_size = ((0, 0), (1, 1)), 
        log_scale = False, amp_data_offset = (0, 0),
        size = (160, 160), resize = False, 
        amp_cmap = 'viridis', phase_cmap = 'hsv', savefigs = False,
        labels = ('x-distance (mm)', 'y-distance (mm)'), dpi = 300,
        plot_lattice = True, lattice_type = slst6_lattice, lattice_offset = (0.866025, 0.0),
        scale = 1.0, unit_cell = 15.0, angle = 0.0, reduce = 20,
        
        animate = False, anim_dpi = 150, anim_lattice = True,
        anim_frames = 120, anim_period = 4, anim_cmap = 'viridis')





"""
datasets.append(
            (freq, 
             'ldsf-disp/ldsf-{}'.format(freq)
             ))

band_structure(datasets, load_cutoff = 5)
"""
"""
#BAND STRUCTURE

datasets = []
args = []
freqs = [6732.9]

size = (220, 220)



kmin = 0
kmax = 81
l = 400

#for x in np.linspace(10000, 11500, 31): freqs.append(x)
#for x in np.linspace(11750, 12300, 12): freqs.append(x)


for freq in freqs:
    amp_data, phase_data = load_lockin_data("../data/ldsf-disp/ldsfd-{}".format(freq), cutoff = 5)     
    real_data = amp_data * np.exp(1j * phase_data / 57.2957795131)

    aspectx = size[0] / len(real_data[0])
    aspecty = size[1] / len(real_data)

    
    real_data = mask_data(amp_data = real_data, 
                                 size = size, 
                                 unit_cell = 30.0, 
                                 aspectx = aspectx, 
                                 aspecty = aspecty, 
                                 real_size = amp_data.shape, 
                                 lattice_offset = (0.5, 0.3333333), 
                                 lattice_type = slst6_lattice, 
                                 angle = 0.0)
    

    kvecs = np.linspace(kmin, kmax, l + 1)
    result = modified_fourier_transform(real_data = real_data, 
                                        real_size = (size[0] / 1000.0, size[1] / 1000.0), 
                                        kvecs = kvecs)
    datasets.append(result)    
    plt.plot(kvecs, result)
    args.append(np.argmax(result)  * (kmax - kmin)/ l + kmin)
    
plt.show()
plt.clf()

plt.imshow(-1.0 * np.array(datasets), cmap = 'gray',
           origin = 'lower', extent = [kmin, kmax, 10000, 12100], rasterized = False, aspect = 'auto')
plt.show()
plt.clf()

plt.scatter(args, freqs)
plt.show()
"""


"""
generate_lattice_mask(size = (200, 200), 
                      unit_cell = 30, 
                      aspect = 1, 
                      real_size = (200,200), 
                      lattice_offset = (0, 0), 
                      lattice_type = slst6_lattice, 
                      angle = 0.0)
"""


"""
amp_data, phase_data = load_lockin_data("../data/ldsf-disp/ldsfh-10000.0", cutoff = 5)     
real_data = amp_data * np.exp(1j * phase_data / 57.2957795131)
    
kvecs = np.linspace(-400, 400, 3201)    
plt.plot(kvecs, modified_fourier_transform(real_data = real_data, 
                                           real_size = (0.230, 0.220), kvecs = kvecs))
"""
