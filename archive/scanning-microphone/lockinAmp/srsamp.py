
import pyvisa as visa
import time
import numpy as np
import pickle

RECORD_DELAY_TIME = 0.015

class LockinAmp(object):
    """
    Implements the interface for the Stanford Research Systems SR830 lock-in amplifier.
    """
    def __init__(self, id = "GPIB::8"):
        rm = visa.ResourceManager('@py')
        potential_device = rm.get_instrument(id)
        potential_device.write("*IDN?\n")
        time.sleep(0.5)  # give the device a bit of time to respond
        test_id = potential_device.read()
        print("Checking out device with ID: %s" % test_id)

        if 'stanford' in test_id.lower():
            print("Connecting to device with ID %s" % test_id)
            self.device = potential_device
            self.name = test_id

            self.device.write("OUTX 1\n")
        else:
            print("This is not a SR830 lock-in amplifier")
            potential_device.close()

    def set_output(self, frequency = 1000, amplitude = 2.5):
        self.device.write("FMOD 1\n")
        self.device.write("FREQ {}\n".format(frequency))
        self.device.write("SLVL {}\n".format(amplitude))

    def initialize_amp(self, sample_rate = 4, time_constant = 5):
        """
        @param sample_rate: The rate at which the lock-in amp samples the signal; see manual
        time constant: 5 means 3ms
        
        
        """
        
        self.device.write("SRAT %d\n" % sample_rate)
        self.device.write("OFLT {}\n".format(time_constant))

    def collect_data(self, data_type = 0):
        """
        @param data_type: What data to query the lock-in amp for; 
                    0 is R - Theta
                    1 is X - Y
        """

        if data_type == 0:
            self.device.write("SNAP? 3,4\n")            
        if data_type == 1:
            self.device.write("SNAP? 1,2\n")

        v1, v2 = [float(value) for value in self.device.read().split(',')]       
        return (v1, v2)

    def record(self, rec_time, delay = RECORD_DELAY_TIME):
        end_time = time.time() + rec_time
        raw_data = []
        while time.time() < end_time:
            try:
                #raw_data.append(self.collect_data())
                raw_data.append(self.device.query_ascii_values('SNAP? 3,4\n', separator = ','))

            except ValueError as v_err:
                print('Got error when fetching amp data: %s' % str(v_err))
                raw_data.append((-1.0, -1.0))
            time.sleep(delay)

        return np.array(raw_data)

    def record_to_file(self, rec_time, filename, delay = RECORD_DELAY_TIME):
        """
        Records <rec_time> seconds of lock-in amp data and saves it as
        a numpy array to the file specified. User does not need to pass in a
        file extension.
        """

        filename += '.pkl'
        frames = self.record(rec_time, delay = delay)              
        frames.dump(filename)
   
 #################################################################################################           

    def _write(self, b):
        self.device.write(b)
    
    def _query(self, b):
        print(self.device.query(b))

    def __repr__(self):
        return "Measurement Device: %s" % self.name


if __name__ == '__main__':
    
    rectime = 2.0
    amp = LockinAmp()
    amp.set_output(frequency = 8000, amplitude = 4.5)
    amp.initialize_amp(sample_rate = 13, time_constant = 4)

    amp._write('SEND 0\n')
    amp._write('REST\n')
    amp._write('STRT\n')
    time.sleep(rectime)
    amp._write('PAUS\n')
    amp.set_output(frequency = 8000, amplitude = 0.05)
    
    start_time = time.time()
    points = int(amp.device.query_ascii_values('SPTS?\n')[0])

    step = 150
    alrd = 0

    amplitudes = []
    phases = []

    while points > alrd + step:
        amp.device.write('TRCA? 1,{},{}\n'.format(alrd, step))
        for value in amp.device.read()[:-2].split(','):
            amplitudes.append(float(value))
        amp.device.write('TRCA? 2,{},{}\n'.format(alrd, step))
        for value in amp.device.read()[:-2].split(','):
            phases.append(float(value))
        alrd += step

    amp.device.write('TRCA? 1,{},{}\n'.format(alrd, points - alrd))
    for value in amp.device.read()[:-2].split(','):
        amplitudes.append(float(value))
    amp.device.write('TRCA? 2,{},{}\n'.format(alrd, points - alrd))
    for value in amp.device.read()[:-2].split(','):
        phases.append(float(value))
    

    print("Collected a total of {} data points, or {} per second".format(points, points/rectime))
    """
    delay = 0.0
    data = amp.record(rec_time = rectime, delay = delay)
    
    amplitudes = [x[0] for x in data]
    phases = [x[1] for x in data]
    """

    amp_mean = np.mean(amplitudes)
    amp_std = np.std(amplitudes)

    phs_mean = np.mean(phases)
    phs_std = np.std(phases)

    #print("Collected {} data points per second".format(float(len(data) / rectime)))
    print('Amplitude data: size {}, mean {}, standard deviation {}, SNR {}'.format(len(amplitudes), amp_mean, amp_std, amp_mean/amp_std))
    print('Phase data: size {}, mean {}, standard deviation {}, SNR {}'.format(len(phases), phs_mean, phs_std, phs_mean / phs_std))

    print('Total read time: {}'.format(time.time() - start_time))

    """
    SNAP method - 29 points / second, SNR ~120, 250
    """