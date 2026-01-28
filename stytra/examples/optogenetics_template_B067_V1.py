import numpy as np
import numpy.typing
from scipy import signal
import datetime

from lightparam import Param

from stytra import Protocol, Stytra
from stytra.stimulation.stimuli import Pause, VisualStimulus, Stimulus
from stytra.stimulation.stimuli.voltage_stimuli import (
    NIVoltageStimulus,
    SetVoltageStimulus,
    VoltagePulseStimulus,
)
from pathlib import Path

import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode
from nidaqmx.constants import LineGrouping, DigitalWidthUnits
from nidaqmx.types import CtrTime

REQUIRES_EXTERNAL_HARDWARE = True

""" IMPORTANT!
The 405 and 488 nm lasers in the Toptica CLE combiner for optogenetics have a bleedthrough (that means
that even if the analog input is 0V, there is still some laser light coming out of the fiber). The
solution is to have a digital enable line as well. This means that there needs to be an analog and a
digital waveform task running on the NI cards.

The Toptica laser software has to be set to:
    - desired laser (405 or 488) needs to be enabled
    - BUT: The emission button needs to be OFF!
    - "TTL Mode" activated
    - "Analog Mode" activated

Assignment of Digital and Analog out channels on Dev2 in B2075:
    - 405 nm
        - Digital Line: p0.0
        - Analog Line: ao2
    - 488 nm
        - Digital Line: p0.1
        - Analog Line: ao3

Assignment of Digital and Analog out channels on Dev2 in B0067:
    - 405 nm
        - Digital Line: p0.0
        - Analog Line: ao2
    - 488 nm
        - Digital Line: p0.1
        - Analog Line: ao3

A laser intensity input of 5V to the Toptica combiner is scaled by the laser intensity
set in the Toptica software: E.g.:
   - 5V input and laser intensity set to 50%: 50% laser output.
   - 1V input and 50% setting: 10% laser output.
"""

"""IMPORTANT!
To avoid damaging the Toptica laser combiner:
The waveforms for the laser intensity and optogenetics galvo
get bundled into one numpy array before being sent to the NI card.
However:

- the Galvos have an allowed control voltage range of -10 to 10V
- BUT: The Toptica laser intensity can only go from 0 to 5V

This means that one has to be extra careful to avoid going above
5V on the analog outputs intended for the laser!
"""

'''
Defining a few example waveforms
'''

def dc(
    samplerate = 100000,        # in samples/second
    waveform_duration = 1,      # in seconds
    value = 1.0               # in Hz
     ):
    '''
    Returns a numpy array with a constant waveform
    '''
    samples =  int(samplerate*waveform_duration)
    return np.full(samples, value)

def sawtooth(
    samplerate = 100000,        # in samples/second
    waveform_duration = 1,      # in seconds
    frequency = 3,              # in Hz
    amplitude = 0,              # in V
    offset = 0,                 # in V
    dutycycle = 50,             # dutycycle in percent
    phase = np.pi/2,            # in rad
    ):
    '''
    Returns a numpy array with a sawtooth function

    Used for creating the galvo signal.

    Example:
    galvosignal =  sawtooth(100000, 0.4, 199, 3.67, 0, 50, np.pi)
    '''

    samples =  int(samplerate*waveform_duration)
    dutycycle = dutycycle/100       # the signal.sawtooth width parameter has to be between 0 and 1
    t = np.linspace(0, waveform_duration, samples)
    # Using the signal toolbox from scipy for the sawtooth:
    waveform = signal.sawtooth(2 * np.pi * frequency * t + phase, width=dutycycle)
    # Scale the waveform to a certain amplitude and apply an offset:
    waveform = amplitude * waveform + offset

    return waveform

def square(
    samplerate = 100000,    # in samples/second
    waveform_duration = 1,  # in seconds
    frequency = 3,          # in Hz
    max_val = 5,          # in V
    min_val = 0,             # in V
    dutycycle = 50,         # dutycycle in percent
    phase = np.pi,          # in rad
    ):
    """
    Returns a numpy array with a rectangular waveform
    """
    amplitude = (max_val - min_val)/2
    offset = (max_val + min_val)/2
    samples =  int(samplerate*waveform_duration)
    dutycycle = dutycycle/100       # the signal.square duty parameter has to be between 0 and 1
    t = np.linspace(0, waveform_duration, samples)

    # Using the signal toolbox from scipy for the sawtooth:
    waveform = signal.square(2 * np.pi * frequency * t + phase, duty=dutycycle)
    # Scale the waveform to a certain amplitude and apply an offset:
    waveform = amplitude * waveform + offset

    return waveform

class NIWaveformStimulus(Stimulus):
    '''
    Generic optogenetics stimulus that consists of:
        - a "gating" digital out
        -

    Writes a

    From the generic stimulus class:
    self.duration
    '''

    def __init__(self, *args, dev="Dev1", do_channels = "p0.0", ao_channels="ao0",  ao_samplerate=100000, ao_min_val=0, ao_max_val=0, ao_waveform, **kwargs):
        self.dev = dev
        self.ao_channels = ao_channels
        self.do_channels = do_channels
        self.ao_min_val = ao_min_val
        self.ao_max_val = ao_max_val
        self.ao_samplerate = ao_samplerate
        self.ao_waveform = ao_waveform
        super().__init__(*args, **kwargs)

    def update(self):
        """Function called by the ProtocolRunner every timestep until the Stimulus
        is over."""
        self.real_time_stop = datetime.datetime.now()

    def handle_interruption(self):
        print("Interruption occured")

    def start(self):
        """Function called by the ProtocolRunner when a new stimulus is set."""

        ''' NI Housekeeping: Analog Out'''
        self.ao_task = nidaqmx.Task()
        self.ao_task.ao_channels.add_ao_voltage_chan(
                                "{}/{}".format(self.dev, self.ao_channels),
                                min_val=self.ao_min_val,
                                max_val=self.ao_max_val,
                                )
        self.ao_task.timing.cfg_samp_clk_timing(rate=self.ao_samplerate,
                                            sample_mode=AcquisitionType.CONTINUOUS)
        #self.actual_sampling_rate = self.ao_task.timing.samp_clk_rate
        ''' NI Housekeeping: Digital Out '''
        self.do_task = nidaqmx.Task()

        self.do_task.do_channels.add_do_chan("{}/{}".format(self.dev, self.do_channels),
                                line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)

        '''
        In order to stop the waveform generation when the protocol is interrupted,
        this connection is necessary. Otherwise, the waveform generation will
        continue. It can still be that the analog waveform stays above 0V if the
        protocol gets interrupted during a higher voltage output (NI cards retain
        their last output voltage when a task stops). However, the digital line
        is low nonetheless, so the laser should be off. This connection can
        only be made after this Stimulus object has been instantiated as the
        self._experiment reference doesn't exist after __init__ and is called later.
        '''
        self._experiment.protocol_runner.sig_protocol_interrupted.connect(self.stop)
        #self._experiment.protocol_runner.sig_protocol_interrupted.connect(self.handle_interruption)

        ''' Enable the laser out line '''
        self.do_task.write([True], auto_start=True)

        self.ao_task.write(self.ao_waveform)
        self.ao_task.start()
        self.real_time_start = datetime.datetime.now()

    def stop(self):
        """ Function called by the ProtocolRunner when a new stimulus is set. """
        try:
            """ Switch off the laser """
            self.do_task.write([False], auto_start=True)

            """ Clean up after yourself """
            self.ao_task.stop()
            #self.do_task.stop()
            self.ao_task.close()
            self.do_task.close()
        except:
            print("Interruption occured, couldn't stop & close NI tasks")

class NIProtocol(Protocol):
    name = "ni_protocol"
    stytra_config = dict(
        camera=dict(type="ximea", camera_params=dict(sn=39314550)),
    )

    def __init__(self):
        super(NIProtocol, self).__init__()
        self.initial_delay = Param(1.0, limits=(0.0, 1000.0), unit="s", loadable=False)
        self.stimulus_duration = Param(
            1.0, limits=(0.0, 1000.0), unit="s", loadable=False
        )
        self.final_delay = Param(1.0, limits=(0.0, 1000.0), unit="s", loadable=False)

        """
        X and Y Voltage Limits are set to correspond to approx. 80% of the FOV of heart camera looking at the
        fish from the side
        """
        self.x_pos_in_volt = Param(0.8, limits=(-3.4, 4.8), unit="V", loadable=False)
        self.y_pos_in_volt = Param(0.8, limits=(-3.0, 4.8), unit="V", loadable=False)
        """
        Intensity limits: Don't use a laser intensity setting larger than 5V with the
        Toptica MLE laser combiners!
        """
        self.intensity_in_volt = Param(2.0, limits=(0.0, 5.0), unit="V", loadable=False)

        # NI Device selection
        self.ni_output_device = "Dev2"
        self.ao_samplerate = 100000
        self.x_channel = "ao0"
        self.y_channel = "ao1"
        self.enable_405_line = "port0/line0"
        self.enable_488_line = "port0/line1"
        self.intensity_channel = "ao2"
        self.ao_channels = "ao0:2"
        self.do_channels = "port0/line0"

        self.update_waveforms()

    def update_protocol(self):
        self.update_waveforms()
        self.super().update_protocol()

    def update_waveforms(self):
        ''' Create a waveform array by combining various waveforms into one numpy array

        Ideally, the analog waveform duration here is 1 second and the truncation of the waveform
        happens via Stytra by having stimulus durations of various intensity - which
        means that these waveforms are a template that can be played for a certain
        amount of time (defined by the stimulus duration in the stimuli list)

        '''

        ''' Example Heart Waveform
        self.intensity_waveform = square(samplerate = self.ao_samplerate,
                                        waveform_duration=1,
                                        frequency=3,
                                        max_val=5,
                                        min_val=0,
                                        dutycycle=10,
                                        phase=np.pi)
        '''
        ''' Example Neuronal Waveform '''
        self.intensity_waveform = square(samplerate = self.ao_samplerate,
                                        waveform_duration=1,
                                        frequency=20,
                                        max_val=5,
                                        min_val=0,
                                        dutycycle=20,
                                        phase=np.pi)

        self.x_waveform = dc(self.ao_samplerate, waveform_duration=1, value=self.x_pos_in_volt)
        self.y_waveform = dc(self.ao_samplerate, waveform_duration=1, value=self.y_pos_in_volt)
        self.merged_waveforms = np.vstack((self.x_waveform, self.y_waveform, self.intensity_waveform))

    def get_stim_sequence(self):
        stimuli = [
            NIWaveformStimulus(
                dev=self.ni_output_device,
                ao_channels=self.ao_channels,
                do_channels=self.do_channels,
                ao_samplerate=self.ao_samplerate,
                ao_min_val=-10,
                ao_max_val=10,
                ao_waveform=self.merged_waveforms,
                duration=self.stimulus_duration,
            ),
            Pause(duration=self.final_delay),
        ]
        return stimuli

if __name__ == "__main__":
    st = Stytra(protocol=NIProtocol())
