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
from nidaqmx.constants import AcquisitionType

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

class OptoWaveformStimulus(Stimulus):
    '''
    Generic optogenetics stimulus

    Writes a

    From the generic stimulus class:
    self.duration
    '''

    def __init__(self, *args, dev="Dev1", channel="ao0",  samplerate=100000, min_val=0, max_val=0, waveform, **kwargs):
        self.dev = dev
        self.channel = channel
        self.min_val = min_val
        self.max_val = max_val
        self.samplerate = samplerate
        self.waveform = waveform
        super().__init__(*args, **kwargs)

    def update(self):
        """Function called by the ProtocolRunner every timestep until the Stimulus
        is over."""
        self.real_time_stop = datetime.datetime.now()

    def start(self):
        """Function called by the ProtocolRunner when a new stimulus is set."""

        ''' NI Housekeeping'''
        self.task = nidaqmx.Task()
        self.task.ao_channels.add_ao_voltage_chan(
                                "{}/{}".format(self.dev, self.channel),
                                min_val=self.min_val,
                                max_val=self.max_val,
                                )
        self.task.timing.cfg_samp_clk_timing(rate=self.samplerate,
                                            sample_mode=AcquisitionType.CONTINUOUS)
        self.actual_sampling_rate = self.task.timing.samp_clk_rate
        ''' '''
        self.task.write(self.waveform)
        self.task.start()
        self.real_time_start = datetime.datetime.now()

    def stop(self):
        """Function called by the ProtocolRunner when a new stimulus is set."""
        self.task.stop()
        self.task.close()



class UpdateableVoltagePulseStimulus(NIVoltageStimulus):
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


class NIProtocol(Protocol):
    name = "ni_protocol"
    stytra_config = dict(
        tracking=dict(method="tail", estimator="vigor"),
        camera=dict(
            video_file=str(Path(__file__).parent / "assets" / "fish_compressed.h5")
        ),
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

class NIProtocol2(Protocol):
    name = "ni_protocol2"
    stytra_config = dict(
        camera=dict(type="ximea", camera_params=dict(sn=39314550)),
    )

    def __init__(self):
        super(NIProtocol2, self).__init__()
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
        self.samplerate = 100000
        self.x_channel = "ao0"
        self.y_channel = "ao1"
        self.intensity_channel = "ao2"
        # Tiny delay for positioning the galvos at the start of the protocol, gets subtracted
        # at the beginning
        self.x_and_y_initial_positioning_time = 0.1

        ''' Create a demo waveform '''
        self.example_intensity_waveform = square(samplerate = self.samplerate,
                                        waveform_duration=1,
                                        frequency=3,
                                        max_val=5,
                                        min_val=0,
                                        dutycycle=10,
                                        phase=np.pi)

    def get_stim_sequence(self):
        stimuli = [
            OptoWaveformStimulus(
                dev=self.ni_output_device,
                channel=self.intensity_channel,
                samplerate=self.samplerate,
                min_val=0,
                max_val=5,
                waveform=self.example_waveform,
                duration=self.stimulus_duration,
            ),
            Pause(duration=self.final_delay),
        ]
        return stimuli


class NIProtocol3(Protocol):
    name = "ni_protocol3"
    stytra_config = dict(
        camera=dict(type="ximea", camera_params=dict(sn=39314550)),
    )

    def __init__(self):
        super(NIProtocol3, self).__init__()
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
        self.samplerate = 100000
        self.x_channel = "ao0"
        self.y_channel = "ao1"
        self.intensity_channel = "ao2"

        self.channels= "ao0:2"


        # Tiny delay for positioning the galvos at the start of the protocol, gets subtracted
        # at the beginning
        self.x_and_y_initial_positioning_time = 0.1
        self.update_waveforms()

    def update_protocol(self):
        self.update_waveforms()
        self.super().update_protocol()

    def update_waveforms(self):
        ''' Create a demo waveform '''
        self.intensity_waveform = square(samplerate = self.samplerate,
                                        waveform_duration=1,
                                        frequency=3,
                                        max_val=5,
                                        min_val=0,
                                        dutycycle=10,
                                        phase=np.pi)
        self.x_waveform = dc(self.samplerate, waveform_duration=1, value=self.x_pos_in_volt)
        self.y_waveform = dc(self.samplerate, waveform_duration=1, value=self.y_pos_in_volt)
        print(self.intensity_waveform.shape)
        print(self.x_waveform.shape)
        print(self.y_waveform.shape)
        self.merged_waveforms = np.vstack((self.x_waveform, self.y_waveform, self.intensity_waveform))
        print(self.merged_waveforms.shape)
        print('Waveforms updated!')

    def get_stim_sequence(self):
        stimuli = [
            OptoWaveformStimulus(
                dev=self.ni_output_device,
                channel=self.channels,
                samplerate=self.samplerate,
                min_val=-10,
                max_val=10,
                waveform=self.merged_waveforms,
                duration=self.stimulus_duration,
            ),
            Pause(duration=self.final_delay),
        ]
        return stimuli

if __name__ == "__main__":
    st = Stytra(protocol=NIProtocol3())
