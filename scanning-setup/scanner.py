"""Main scanning routine that uses both the abstracted microphone class and
the printer class to scan and record audio samples. The resultant data may
have to be processed differently depending on what data source is used. The
current data sources have been:
    * raw FFT samples from an Oscilloscope
    * raw audio waveform from a microphone
The first one takes much less time since the FFT is being done onboard, so
usually we will be using raw FFT samples from the attached oscilloscope.
"""
import time
import sys
import numpy as np
import os
import tqdm
sys.path.append('./microphone')
sys.path.append('./printer')

from matplotlib import pyplot as plt
import math

# Switch up the import depending on which data collection device you're using
# from microphone import Microphone
#from oscilloscope import OscilloscopeMicrophone as Microphone
from printer import Printer
from srsamp import LockinAmp
#from siggen import SignalGenerator


PRINTER_CONNECT_TIME = 10
MOVEMENT_DELAY_TIME = 0.2
MOVEMENT_DELAY_MULTIPLIER = 0.1
scan_points_multiplier = 10.0

slst6_lattice = [#(( 0.0000,  0.0000), 0.4),
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

extkag_lattice = [((-0.3660,  0.0000), 0.15),
                ((-0.1830,  0.3170), 0.15),
                (( 0.1830,  0.3170), 0.15),
                (( 0.3660,  0.0000), 0.15),
                (( 0.1830, -0.3170), 0.15),
                ((-0.1830, -0.3170), 0.15),
                (( 0.0000,  0.6340), 0.15),
                (( 0.5490,  0.3170), 0.15),
                (( 0.5490, -0.3170), 0.15),
                (( 0.0000, -0.6340), 0.15),
                ((-0.5490, -0.3170), 0.15),
                ((-0.5490,  0.3170), 0.15),
                (( 0.0000,  1.0000), 0.15),
                (( 0.8660,  0.5000), 0.15),
                (( 0.8660, -0.5000), 0.15),
                (( 0.0000, -1.0000), 0.15),
                ((-0.8660, -0.5000), 0.15),
                ((-0.8660,  0.5000), 0.15)]

## ADJUST THESE LATER
#dir_scan_pos = [-275, 0, 10]
#freq_sweep_pos = [0,0,0]

def inside(x, y, area):
    if x < 0: return False
    if y < 0: return False
    if x > area[0]: return False
    if y > area[1]: return False

    return True

def sort_lattice_sites(candidates, cutoff):
    
    result = []
    previous = (0.0, 0.0)

    while len(candidates) > 0:
        distances = [((point[0] - previous[0])**2 + (point[1] - previous[1])**2) for point in candidates]
        
        previous = candidates[np.argmin(distances)]
        
        if np.array(distances).min() > cutoff:
            result.append(previous)

        candidates.remove(previous)
    
    return result


class Scanner(object):
    """Scanner object that manages the printer and the microphone. Each object
    should represent any sequence of scans using the same microphone and
    printer"""
    
    def __init__(self, serial = None):
	
        self.amp = LockinAmp()
        self.amp.initialize_amp()
	
        if not serial:
            serial = '/dev/ttyACM0'        
        print('Trying to connect printer through USB port {}'.format(serial))        
        self.p = Printer(serial = serial)

        # Wait some time for handshake to occur with printer
        time.sleep(PRINTER_CONNECT_TIME)

        if self.p.online():
            print("Printer connected!")
        else:
            raise RuntimeError("Printer is not online. Are you connecting to right USB?")
    
    def move_speed(self, x = None, y = None, z = None, speed = 500, delay = 0.2):
        """
        TODO: FIX THIS FUNCTION YOU DOOFUS
        Move to coordinate at a certain speed. Will calculate the delay time
        needed for the program to wait during travel time by estimating the
        distance traveled and the speed.
        @param x (int): distance to travel in x axis (mm) 
        @param y (int): distance to travel in y axis (mm) 
        @param z (int): distance to travel in z axis (mm) 
        @param speed (int): speed of nozzle in mm/min
        """
        speed_per_second = float(speed) / 60.0
        dx = 0.0 if not x else x
        dy = 0.0 if not y else y
        dz = 0.0 if not z else z
        move_vector = np.array([dx, dy, dz])
        distance = np.sqrt((move_vector ** 2).sum())

        # Actually send the move command, overriding any previous command
        self.p.move_coord(x=x, y=y, z=z, speed=speed)
        # Sleep the amount of time while the nozzle is moving, plus a small
        # delay for tolerance reasons
        time.sleep(distance / speed_per_second + delay)
    
    def move_speed_noblock(self, x = None, y = None, z = None, speed = 500, delay = 0.2):
        """
        Move to coordinate at a certain speed. Will calculate the delay time
        needed for the program to wait during travel time by estimating the
        distance traveled and the speed. Will not block the main program
        while moving the nozzle; will return the time it will take to actually
        travel the distance. Allows us to record from the microphone as we are
        scanning this distance.
        @param x (int): distance to travel in x axis (mm) 
        @param y (int): distance to travel in y axis (mm) 
        @param z (int): distance to travel in z axis (mm) 
        @param speed (int): speed of nozzle in mm/min
        """
        speed_per_second = float(speed) / 60.0
        dx = 0.0 if not x else x
        dy = 0.0 if not y else y
        dz = 0.0 if not z else z
        move_vector = np.array([dx, dy, dz])
        distance = np.sqrt((move_vector ** 2).sum())

        # Actually send the move command, overriding any previous command
        self.p.move_coord(x=x, y=y, z=z, speed=speed)
        # Sleep the amount of time while the nozzle is moving, plus a small
        # delay for tolerance reasons
        return distance / speed_per_second

    def set_as_origin(self):
        # TODO: Set the current location as the origin (0, 0, 0) for the
        # scanner. The move seems to be a bit buggy with this though.
        raise NotImplementedError()
   
    def __str__(self):
        return "Scanner object"

    def __repr__(self):
        status = "Online" if self.p.online() else "Offline"
        return "Scanner [%s]" % status

    def move(self, x = None, y = None, z = None, delay = MOVEMENT_DELAY_TIME, delay_factor = MOVEMENT_DELAY_MULTIPLIER):
        """
        Displaces the head by x, y, z units. Will find the shortest distances to get to the
        endpoint by moving stepper motors simultaneously. Delay will pause control sequence to give
        the CNC time to move to the location, since we are NOT BLOCKING!
        """

        if not self.p.online():
            raise RuntimeError("Cannot move - printer is not connected.")
        
        self.p.move_coord(x = x, y = y, z = z)
        # Will sleep for a factor of the amount of distance we need to travel
        dx = 0 if not x else x
        dy = 0 if not y else y
        dz = 0 if not z else z
        distance = (dx**2 + dy**2 + dz**2) ** 0.5
        time.sleep(delay + distance * delay_factor)

    def scan_continuous_lattice(self, distance_x, distance_y, y_resolution, frequency, scan_speed = 1500,
        delay = 0.5, amplitude = 2.5, savepath="./data", title = None, note=""):
        """
        Scans lines across the x axis, with steps happening along the y axis.
        If we have a rectangular region, the scan lines will look like:

                |-------- x distance ----|
                __________________________
                __________________________
                ...
                __________________________

        When moving in a straight line, the microphone will be polled continuously
        during the time, so we can just linearly interpolate the location of
        the microphone at any given time here. The format of the saved files
        will be:
            <savepath>/<title>/continuous_<xmin>_<xmax>_<yloc>.wav
        @param resolution: number of samples for the y dimension. For example,
            if we scan a 100 mm x 100 mm box, a scan size of 51 will mean
            a line every 2 mm on the y dimension
        @param savepath: folder that your saved files will be sent to.

        """

        start_time = time.time()
        if not title:
            title = "lds-" + str(start_time)

        # Create a folder to store all of our sound samples in
        savefolder = os.path.join(savepath, title)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        print("Scanning at a frequency of {} Hz".format(frequency))

        self.amp.set_output(frequency = frequency, amplitude = amplitude)
        time.sleep(delay)

        with open(os.path.join(savefolder, 'info'), 'w') as f:
            f.write("Scanning on grid, continuously polling. Using parameters:\n")
            f.write('Frequency: %d\n' % frequency)
            f.write('Amplitude: %d\n' % amplitude)
            f.write('End Coordinates: %s\n' % str((distance_x, distance_y)))
            f.write('Y Resolution: %d\n' % y_resolution)
            f.write('Scan Speed: %d\n' % scan_speed)
            f.write("\n\n")
            f.write("Additional notes:\n%s\n" % note)

        print("Expected Number of samples per line: %d" % int((distance_x * scan_points_multiplier)/ scan_speed))
        
        ########### here i am, rock you like a hurricane
        
        scan_points = []

        for y in np.linspace(0, distance_y, y_resolution):
            if (len(scan_points) / 2) % 2 == 0:
               scan_points.append((0.0, y))
               scan_points.append((distance_x, y))
            else:
               scan_points.append((distance_x, y))
               scan_points.append((0.0, y))

        scan_points = scan_points[1:]
        
        # The even indices will be scanning/moving, while the odd indices
        # will be moving only. 

        previous_coord = (0, 0)
        for idx, coord in tqdm.tqdm(list(enumerate(scan_points))):
            p_x, p_y = coord
            dx = p_x - previous_coord[0]
            dy = p_y - previous_coord[1]

            if idx % 2 == 0:
                fname = os.path.join(savefolder, "continuous_0_{}_{}".format(dx, p_y))
                record_time = self.move_speed_noblock(x = dx, y = dy, speed = scan_speed)
                self.amp.record_to_file(rec_time = record_time, filename = fname)
                time.sleep(delay)  # some padding time
            else:
                self.move_speed(x = dx, y = dy, speed = scan_speed)
                time.sleep(delay)  # some padding time
            
            previous_coord = (p_x, p_y)

        # Move back to our original location. Important since we are using relative coordinates.

        if ((len(scan_points) + 1) / 2) % 2 == 0:
            distance_x = 0.0

        self.move(x = -distance_x, y = -distance_y)
        
        self.amp.set_output(frequency = 100, amplitude = 0.1)
        time.sleep(delay)

        end_time = time.time()
        self.p.save_position_to_file()
        print('Total Scan Time: %s s' % str(end_time - start_time))

    def point_chirp_to_file(self, freqs, filename, sg_delay = 0.25, relax = 0.05, amplitude = 2.5):
        data = []
        
        for freq in freqs:
            self.amp.set_output(frequency = freq, amplitude = amplitude)
            time.sleep(sg_delay)

            self.amp.device.write("SNAP? 3,4\n") 
            amp, phase = [float(value) for value in self.amp.device.read().split(',')]             
            
            data.append((freq, amp, phase))
            time.sleep(relax)

        self.amp.set_output(frequency = 1500, amplitude = amplitude)
        np.array(data).dump(filename)

    def find_lattice_sites(self, area, unit_cell, lattice_offset, angle, lattice_type):
        candidates = []

        for coord in lattice_type:
            for xi in range(-1 * (int(area[0] / unit_cell) + 5), (int(area[0] / unit_cell) + 5)):
                for yi in range(-1 * (int(area[1] / unit_cell) + 5), (int(area[1] / unit_cell) + 5)):

                    xl = (coord[0][0] + xi * math.sqrt(3) + lattice_offset[0]) * unit_cell
                    yl = (coord[0][1] + yi * 3 + lattice_offset[1]) * unit_cell

                    xr = xl * math.cos(angle) - yl * math.sin(angle)
                    yr = xl * math.sin(angle) + yl * math.cos(angle)

                    if inside(xr, yr, area):
                        candidates.append((xr, yr))

                       ###

                    xl = (coord[0][0] + (xi - 0.5) * math.sqrt(3) + lattice_offset[0]) * unit_cell
                    yl = (coord[0][1] + (yi + 0.5) * 3 + lattice_offset[1]) * unit_cell

                    xr = xl * math.cos(angle) - yl * math.sin(angle)
                    yr = xl * math.sin(angle) + yl * math.cos(angle)

                    if inside(xr, yr, area):
                        candidates.append((xr, yr))

        # SORT THEM

        return candidates
    
    def site_chirp_scan(self, area, min_freq, max_freq, sample_no, lattice_offset,
                        unit_cell = 30.0, amplitude = 3.5, g_delay = 0.5, sg_delay = 0.005, move_delay = 0.5,                      
                        angle = 0.0, lattice_type = slst6_lattice, speed = 2500, title = None, 
                        savepath = "./data", note = '', relax = 0.0, gdz = 2.50, descent_delay = 1.0):
        
        start_time = time.time()
        if not title:
            title = "scp-" + str(start_time)

        # Create a folder to store all of our sound samples in
        savefolder = os.path.join(savepath, title)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        print('Scanning via chirp!')

        self.amp.set_output(frequency = 2000, amplitude = 1.5)
        time.sleep(g_delay)

        with open(os.path.join(savefolder, 'info'), 'w') as f:
            f.write("Scanning at small sites, chirping. Using parameters:\n")
            f.write('Frequency range: {} - {} Hz, {} samples\n'.format(min_freq, max_freq, sample_no))
            f.write('Amplitude: %d\n' % amplitude)
            f.write('End Coordinates: %s\n' % str(area))
            f.write('Move Speed: %d\n' % speed)
            f.write("\n\n")
            f.write("Additional notes:\n%s\n" % note)
        
        #####       
        
        scan_points = sort_lattice_sites(self.find_lattice_sites(area = area, 
                                            unit_cell = unit_cell, 
                                            lattice_offset = lattice_offset, 
                                            angle = angle, 
                                            lattice_type = lattice_type),
                                            cutoff = (lattice_type[0][1] / 2.0)**2 )
        
        """
        plt.scatter([p[0] for p in scan_points], [p[1] for p in scan_points])
        plt.show()
        """
        
        freq_range = np.linspace(min_freq, max_freq, sample_no) 

        previous = (0.0, 0.0)


        for index, coord in tqdm.tqdm(list(enumerate(scan_points))):
            x, y = coord
            dx = x - previous[0]
            dy = y - previous[1]

            self.move_speed(x = dx, y = dy, speed = speed)
            time.sleep(move_delay)  # some padding time

            
            self.move_speed(z = -gdz, speed = speed)
            time.sleep(move_delay)  # some padding time
            time.sleep(descent_delay)
            
            
            filename = os.path.join(savefolder, "scppos_{}_{}.pkl".format(x, y))
            self.point_chirp_to_file(freqs = freq_range, 
                                    filename = filename, 
                                    sg_delay = sg_delay, 
                                    relax = relax, 
                                    amplitude = amplitude)
            
            
            self.move_speed(z = gdz, speed = speed)
            time.sleep(move_delay)  # some padding time
            
            

            previous = (x, y)

        self.move(x = -previous[0], y = -previous[1])
        time.sleep(g_delay)

        end_time = time.time()
        #self.p.save_position_to_file()
        print('Total Scan Time: %s s' % str(end_time - start_time))
        
###################################################################################################################

    def scan_cstm_lattice(self, distance_x, distance_y, x_resolution, y_resolution,
                                 record_time=2.0, savepath="./data",speed=2500):
        """Scans along a square lattice and saves each audio clip at each location.
        Audio clips will be saved the format:
            <savepath>/<time.time()>_<xloc>_<yloc>_<zloc>.wav
        @param begin_coord: tuple of x and y coordinate to start out with. Since the current code
                            is location agnostic, it is assume the CNC is over begin_coord already
                            TODO: Write a re-centering script or something lmao
        @param end_coord: tuple of x and y coordinate to scan until
        @param resolution: number of samples for each dimension. If 10 is selected, we'll scan 100 points.
        @param savepath: folder that your saved wave files will be sent to.
        """
        # Create a folder to store all of our sound samples in
        print_begin_time = int(time.time())
        savefolder = os.path.join(savepath, "v2-" + str(print_begin_time))
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        # since we are assuming that we start at the begin_coord, consider the relative coordinates where
        # begin_coord is just the origin already.
        scan_points = [(x, y) for x in np.linspace(0, distance_x, x_resolution) 
                              for y in np.linspace(0, distance_y, y_resolution)]
        
        # Beginning at the begin_coord, we are doing to stop and keep scanning
        previous_coord = (0,0)
        for p_x, p_y in tqdm.tqdm(scan_points):
            dx = p_x - previous_coord[0]
            dy = p_y - previous_coord[1]
            self.move(x=dx, y=dy,speed=speed)
            fname = os.path.join(savefolder, "{}_{}_{}".format(p_x, p_y, 0))
            self.mic.record_to_file(record_time, fname)
            previous_coord = p_x, p_y

        # Move back to our original location. Important since we are using relative coordinates.
        self.move(x=-distance_x, y=-distance_y)
    
    def freq_transmission_sweep(self, freq_start, freq_end, n_samples, distance = 230, move_speed = 4000, 
                scan_speed = 500, delay = 0.1, bandwidth = 200, maxf = 50000, ns = 10000, ptime=0.5,
                amplitude = 10, savepath="./data", note=""):

        """
        Scans a single line across the y-axis, at n_samples different frequencies evenly spaced 
        between freq_start and freq_end. The microphone will be polled continuously during the time, 
        so we can just linearly interpolate the location of the microphone at any given time here. 
        The format of the saved files will be:
            <savepath>/<time.time()>/freqsweep_<ymin>_<ymax>_<freq>.wav
        @param distance: the length of the portion to be scanned along the y-axis 
        @param scan_speed: determines the spatial resolution of the scan
        @param savepath: folder that your saved wave files will be sent to.
        """

        start_time = time.time()
        # Create a folder to store all of our sound samples in
        print_begin_time = int(time.time())
        savefolder = os.path.join(savepath, "freqsweep-" + str(print_begin_time))
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        with open(os.path.join(savefolder, 'info'), 'w') as f:
            f.write("Scanning at the end of metamaterial opposite from transducer, using parameters: \n")
            f.write('Frequency start: %d\n' % freq_start)
            f.write('Frequency end: %d\n' % freq_end)
            f.write('Bandwidth: %d\n' % bandwidth)
            f.write('No. of samples: %d\n' % n_samples)
            f.write('Distance: %d\n' % distance)
            f.write('Scan speed: %d\n' % scan_speed)
            f.write('Amplitude: %d\n' % amplitude)
            f.write('Height: %d\n' % self.p.z_pos)
            f.write("\n\n")

            f.write("Additional notes:\n%s\n" % note)
        
        time.sleep(2)

        if not self.siggen:
            self.siggen = SignalGenerator()
            print("Initializing signal generator...")
            time.sleep(10)
        
        print("Expected number of points: %d" % int((distance * 374.0)/ scan_speed))

        #############
        #count=-1

        for freq in np.linspace(freq_start, freq_end, n_samples):
            
            ###############
            #count+=1

            sample_start = np.floor((freq - bandwidth/ 2.0) / (maxf/ns)) 
            sample_end =  np.ceil((freq + bandwidth/ 2.0) / (maxf/ns))

            print("Scanning around a frequency of {} Hz, at samples {} to {}".format(
                freq, sample_start, sample_end))

            self.siggen.set_frequency(freq, amplitude = amplitude, offset = 0)
            
            time.sleep(ptime)

            ####################INSERT COUNT HERE IF REPEATING FREQUENCY############################
            fname = os.path.join(savefolder, "fs_0_{}_{}".format(distance, freq))
            record_time = self.move_speed_noblock(x = distance, speed = scan_speed)
            self.mic.record_to_file(record_time, fname, delay=delay,
                sample_start=sample_start, sample_end=sample_end)
            time.sleep(ptime)  # some padding time
            self.move_speed(x = -distance, speed=move_speed)
            time.sleep(ptime)  # some padding time
            ###############################
            #time.sleep(1)
            #self.p.move_coord(z = 1)
            #time.sleep(1)


        end_time = time.time()
        self.p.save_position_to_file()
        print('Total Scan Time: %s s' % str(end_time - start_time))

        self.siggen.set_frequency(1, amplitude = 0.01, offset = 0)

        
        
        ############################################################################################
        """
        
        scan_points = [[(0, y), (distance_x, y)] for y in np.linspace(0, distance_y, y_resolution)]
        scan_points = [item for sublist in scan_points for item in sublist]
        scan_points = scan_points[1:]
        
        # Beginning at the begin_coord, we are doing to stop and keep scanning
        # The even indices will be scanning/moving, while the odd indices
        # will be moving only.
 

        previous_coord = (0, 0)
        for idx, coord in tqdm.tqdm(list(enumerate(scan_points))):
            p_x, p_y = coord
            dx = p_x - previous_coord[0]
            dy = p_y - previous_coord[1]

            if idx % 2 == 0:
                

        # Move back to our original location. Important since we are using relative coordinates.
        self.move(x=-distance_x, y=-distance_y)
        
        """

    def freq_bp_sweep(self, freq_start, freq_end, n_samples, distance = 25, speed = 500, delay = 0.1, 
                        bandwidth = 100, maxf = 50000, ns = 10000, ptime=0.5,
                        amplitude = 10, savepath="./data", note=""):

        """
        NO, NEED TO CORRECT
        Scans a single line across the y-axis, at n_samples different frequencies evenly spaced 
        between freq_start and freq_end. The microphone will be polled continuously during the time, 
        so we can just linearly interpolate the location of the microphone at any given time here. 
        The format of the saved files will be:
            <savepath>/<time.time()>/freqsweep_<ymin>_<ymax>_<freq>.wav
        @param distance: the length of the portion to be scanned along the y-axis 
        @param scan_speed: determines the spatial resolution of the scan
        @param savepath: folder that your saved wave files will be sent to.
        """

        start_time = time.time()
        # Create a folder to store all of our sound samples in
        print_begin_time = int(time.time())
        savefolder = os.path.join(savepath, "bpfs-" + str(print_begin_time))
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        with open(os.path.join(savefolder, 'info'), 'w') as f:
            f.write("Scanning at the end of metamaterial opposite from transducer, using parameters: \n")
            f.write('Frequency start: %d\n' % freq_start)
            f.write('Frequency end: %d\n' % freq_end)
            f.write('Bandwidth: %d\n' % bandwidth)
            f.write('No. of samples: %d\n' % n_samples)
            f.write('Distance: %d\n' % distance)
            f.write('Scan speed: %d\n' % speed)
            f.write('Amplitude: %d\n' % amplitude)
            f.write('Height: %d\n' % self.p.z_pos)
            f.write("\n\n")

            f.write("Additional notes:\n%s\n" % note)
        
        time.sleep(2)

        if not self.siggen:
            self.siggen = SignalGenerator()
            print("Initializing signal generator...")
            time.sleep(10)
        
        print("Expected number of points: %d" % int((distance * 374.0)/ speed))

        #############
        #count=-1

        for freq in np.linspace(freq_start, freq_end, n_samples):
            
            ###############
            #count+=1

            sample_start = np.floor((freq - bandwidth/ 2.0) / (maxf/ns)) 
            sample_end =  np.ceil((freq + bandwidth/ 2.0) / (maxf/ns))

            print("Scanning around a frequency of {} Hz, at samples {} to {}".format(
                freq, sample_start, sample_end))

            self.siggen.set_frequency(freq, amplitude = amplitude, offset = 0)
            
            time.sleep(ptime)

            ####################INSERT COUNT HERE IF REPEATING FREQUENCY############################
            fname = os.path.join(savefolder, "fs_0_{}".format(freq))
            record_time = self.move_speed_noblock(y = distance, speed = speed)
            self.mic.record_to_file(record_time, fname, delay=delay,
                sample_start=sample_start, sample_end=sample_end)
            time.sleep(ptime)  # some padding time

            distance *= (-1.0)
            ###############################
            #time.sleep(1)
            #self.p.move_coord(z = 1)
            #time.sleep(1)


        end_time = time.time()
        self.p.save_position_to_file()
        print('Total Scan Time: %s s' % str(end_time - start_time))

        self.siggen.set_frequency(1, amplitude = 0.01, offset = 0)

    def freq_above_sweep(self, freq_start, freq_end, n_samples, delay = 0.1, bandwidth = 200, maxf = 50000, ns = 10000, ptime=0.5,
                amplitude = 10, savepath="./data", note="", record_time = 2.0):

        """
        Scans above a single point, at n_samples different frequencies evenly spaced 
        between freq_start and freq_end. The microphone will be polled continuously during the time.
        The format of the saved files will be:
            <savepath>/<time.time()>/abfreqsweep_<ymin>_<ymax>_<freq>.wav
        @param savepath: folder that your saved wave files will be sent to.
        """

        start_time = time.time()
        # Create a folder to store all of our sound samples in
        print_begin_time = int(time.time())
        savefolder = os.path.join(savepath, "ab-freqsweep-" + str(print_begin_time))
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        with open(os.path.join(savefolder, 'info'), 'w') as f:
            f.write("Scanning above single point of the metamaterial, using parameters: \n")
            f.write('Frequency start: %d\n' % freq_start)
            f.write('Frequency end: %d\n' % freq_end)
            f.write('Bandwidth: %d\n' % bandwidth)
            f.write('No. of samples: %d\n' % n_samples)
            f.write('Amplitude: %d\n' % amplitude)
            f.write('Height: %d\n' % self.p.z_pos)
            f.write("\n\n")

            f.write("Additional notes:\n%s\n" % note)
        
        time.sleep(2)

        if not self.siggen:
            self.siggen = SignalGenerator()
            print("Initializing signal generator...")
            time.sleep(10)
        
        #############
        #count=-1

        for freq in np.linspace(freq_start, freq_end, n_samples):
            
            ###############
            #count+=1

            sample_start = np.floor((freq - bandwidth/ 2.0) / (maxf/ns)) 
            sample_end =  np.ceil((freq + bandwidth/ 2.0) / (maxf/ns))

            print("Scanning around a frequency of {} Hz, at samples {} to {}".format(
                freq, sample_start, sample_end))

            self.siggen.set_frequency(freq, amplitude = amplitude, offset = 0)
            
            time.sleep(ptime)

            ####################INSERT COUNT HERE IF REPEATING FREQUENCY############################
            fname = os.path.join(savefolder, "abfs_{}".format(freq))
            self.mic.record_to_file(record_time, fname, delay=delay,
                sample_start=sample_start, sample_end=sample_end)
            time.sleep(ptime)  # some padding time
            ###############################
            #time.sleep(1)
            #self.p.move_coord(z = 1)
            #time.sleep(1)


        end_time = time.time()
        self.p.save_position_to_file()
        print('Total Scan Time: %s s' % str(end_time - start_time))

        self.siggen.set_frequency(1, amplitude = 0.01, offset = 0)



if __name__ == '__main__':
    print("Please use the main.py script for running scans!")
    
    scan = Scanner()
    time.sleep(3)
    
    
    scan.site_chirp_scan(area = (30, 30), min_freq = 2000, max_freq = 4000, 
                        sample_no = 6, lattice_offset = (-0.5490, -0.3170),
                        unit_cell = 20.0, amplitude = 4.0, g_delay = 0.5,                        
                        angle = 0.0, lattice_type = extkag_lattice, speed = 2000, title = None, 
                        savepath = "./data", note = '', relax = 0.0, gdz = 3.0)

    """
    scan.site_chirp_scan(area = (220, 220), min_freq = 11200, max_freq = 11500, 
                        sample_no = 451, lattice_offset = (0.866025, 0.0),
                        unit_cell = 30.0, amplitude = 3.5, g_delay = 0.5,                        
                        angle = 0.0, lattice_type = slst6_lattice, speed = 1500, title = None, 
                        savepath = "./data", note = '', relax = 0.0)
    
    scan.site_chirp_scan(area = (220, 220), min_freq = 6700, max_freq = 6760, 
                        sample_no = 61, lattice_offset = (0.866025, 0.0),
                        unit_cell = 30.0, amplitude = 3.5, g_delay = 0.5,                        
                        angle = 0.0, lattice_type = slst6_lattice, speed = 1500, title = None, 
                        savepath = "./data", note = '', relax = 0.0)

    scan.site_chirp_scan(area = (220, 220), min_freq = 6700, max_freq = 6760, 
                        sample_no = 61, lattice_offset = (0.866025, 0.0),
                        unit_cell = 30.0, amplitude = 4.5, g_delay = 0.5,                        
                        angle = 0.0, lattice_type = slst6_lattice, speed = 1500, title = None, 
                        savepath = "./data", note = '', relax = 0.0)
    """
    """
    scan.site_chirp_scan(area = (220, 220), min_freq = 11448, max_freq = 11468, 
                        sample_no = 401, lattice_offset = (0.866025, 0.0),
                        unit_cell = 30.0, amplitude = 4.5, g_delay = 0.5,                        
                        angle = 0.0, lattice_type = slst6_lattice, speed = 1500, title = None, 
                        savepath = "./data", note = '', relax = 0.0)
    
    scan.site_chirp_scan(area = (220, 220), min_freq = 5400, max_freq = 5600, 
                        sample_no = 201, lattice_offset = (0.866025, 0.0),
                        unit_cell = 30.0, amplitude = 4.5, g_delay = 0.5,                        
                        angle = 0.0, lattice_type = slst6_lattice, speed = 1500, title = None, 
                        savepath = "./data", note = '', relax = 0.0)
    
    
    scan.site_chirp_scan(area = (220, 220), min_freq = 5502.5, max_freq = 15502.5, 
                        sample_no = 2001, lattice_offset = (0.866025, 0.0),
                        unit_cell = 30.0, amplitude = 4.5, g_delay = 0.5,                        
                        angle = 0.0, lattice_type = slst6_lattice, speed = 1500, title = None, 
                        savepath = "./data", note = '', relax = 0.0)
    """
        

    "0.166 seconds per site per sample  83/s"

    """
    scan.amp.set_output(frequency = 2000, amplitude = 3.75)

    amps = []
    phases = []
    times = np.linspace(0.005, 0.105, 6)

    for sgd in times:
        amp, phase = scan.point_chirp(np.linspace(2000, 4000, 26), sg_delay = 0.005, relax = 0.0, amplitude = 3.75)
        amps.append(amp)
        phases.append(phase)

    scan.amp.set_output(frequency = 100, amplitude = 0.1)

    for i in range(len(amps)):
        plt.plot(amps[i])
    
    plt.show()
    plt.clf()

    for i in range(len(phases)):
        plt.plot(phases[i])
    plt.show()
    plt.clf()
    """
    
    
