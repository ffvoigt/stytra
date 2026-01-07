from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from lightparam import Param

from stytra import Stytra
from stytra.experiments.fish_pipelines import HeartRatePipeline
from stytra.gui.camera_display import CameraSelection, HeartRateSelection
from stytra.stimulation import Protocol
from stytra.stimulation.stimuli import MovingGratingStimulus, Pause
from stytra.tracking.heart import HeartRateMethod

REQUIRES_EXTERNAL_HARDWARE = False


class HeartRateProtocol(Protocol):
    name = "heart_rate_display"

    stytra_config = dict(
        tracking=dict(method=HeartRatePipeline),
        camera=dict(type="ximea", camera_params=dict(sn=39310050)),
    )

    def __init__(self):
        super().__init__()
        self.total_duration_sec = Param(10.0, limits=(0.2, None))

    def get_stim_sequence(self):
        stimuli = [
            Pause(duration=self.total_duration_sec),
        ]
        return stimuli


if __name__ == "__main__":
    s = Stytra(protocol=HeartRateProtocol())
