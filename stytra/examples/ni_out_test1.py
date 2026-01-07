from lightparam import Param

from stytra import Protocol, Stytra
from stytra.stimulation.stimuli import Pause
from stytra.stimulation.stimuli.voltage_stimuli import (
    NIVoltageStimulus,
    SetVoltageStimulus,
)

REQUIRES_EXTERNAL_HARDWARE = True


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
        self.x_pos_in_volt = Param(1.0, limits=(-10.0, 10.0), unit="V", loadable=False)
        self.y_pos_in_volt = Param(1.0, limits=(-10.0, 10.0), unit="V", loadable=False)
        self.intensity_in_volt = Param(1.0, limits=(0.0, 5.0), unit="V", loadable=False)
        self.ni_output_device = "Dev2"
        self.x_channel = "ao0"
        self.y_channel = "ao1"
        self.intensity_channel = "ao2"

    def get_stim_sequence(self):
        stimuli = [
            SetVoltageStimulus(
                dev=self.ni_output_device,
                chan=self.x_channel,
                min_val=-10,
                max_val=10,
                voltage=self.x_pos_in_volt,
            ),
            SetVoltageStimulus(
                dev=self.ni_output_device,
                chan=self.y_channel,
                min_val=-10,
                max_val=10,
                voltage=self.y_pos_in_volt,
            ),
            Pause(duration=self.initial_delay),
            SetVoltageStimulus(
                dev=self.ni_output_device,
                chan=self.intensity_channel,
                min_val=0,
                max_val=5,
                voltage=self.intensity_in_volt,
            ),
            Pause(duration=self.stimulus_duration),
            SetVoltageStimulus(
                dev=self.ni_output_device,
                chan=self.intensity_channel,
                min_val=0,
                max_val=5,
                voltage=0,
            ),
            Pause(duration=self.final_delay),
        ]
        return stimuli


if __name__ == "__main__":
    st = Stytra(protocol=NIProtocol())
