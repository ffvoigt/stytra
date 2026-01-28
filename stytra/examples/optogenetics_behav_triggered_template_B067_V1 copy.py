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

class UpdateableVoltagePulseStimulus(NIVoltageStimulus):
    """

    """

    def __init__(self, *args, high=1.0, low=0.0, **kwargs):
        self.high = high  # high voltage of the pulse
        self.low = low  # low voltage of the pulse
        # self.duration = duration
        super().__init__(*args, **kwargs)

    def update(self):
        fish_vel = self._experiment.estimator.get_velocity()

        if fish_vel < -7:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(
                    "{}/{}".format(self.dev, self.chan),
                    min_val=self.min_val,
                    max_val=self.max_val,
                )
                task.write(self.high)
        else:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(
                    "{}/{}".format(self.dev, self.chan),
                    min_val=self.min_val,
                    max_val=self.max_val,
                )
                task.write(self.low)

    def stop(self):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(
                "{}/{}".format(self.dev, self.chan),
                min_val=self.min_val,
                max_val=self.max_val,
            )
            task.write(self.low)

class BehaviorTriggeredNIProtocol(Protocol):
    name = "ni_protocol"
    stytra_config = dict(
        tracking=dict(method="tail", estimator="vigor"),
        camera=dict(
            video_file=str(Path(__file__).parent / "assets" / "fish_compressed.h5")
        ),
    )

    def __init__(self):
        super(BehaviorTriggeredNIProtocol, self).__init__()
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
        self.x_channel = "ao0"
        self.y_channel = "ao1"
        self.intensity_channel = "ao2"
        # Tiny delay for positioning the galvos at the start of the protocol, gets subtracted
        # at the beginning
        self.x_and_y_initial_positioning_time = 0.1

    def get_stim_sequence(self):
        stimuli = [
            UpdateableVoltagePulseStimulus(
                dev=self.ni_output_device,
                chan=self.intensity_channel,
                min_val=0.0,
                max_val=5.0,
                high=5.0,
                low=0.0,
                duration=self.stimulus_duration,
            ),
        ]
        return stimuli

if __name__ == "__main__":
    st = Stytra(protocol=BehaviorTriggeredNIProtocol())
