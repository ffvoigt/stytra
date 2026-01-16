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

REQUIRES_EXTERNAL_HARDWARE = True



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


if __name__ == "__main__":
    st = Stytra(protocol=NIProtocol())
