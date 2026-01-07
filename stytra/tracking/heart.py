import cv2
import numpy as np
from numba import float64, int64, jit

try:
    from numba.experimental import jitclass
except ModuleNotFoundError:
    from numba import jitclass

from collections import namedtuple
from itertools import chain

from lightparam import Param
from scipy.fft import fft, fftfreq

from stytra.tracking.pipelines import ImageToDataNode, NodeOutput
from stytra.tracking.simple_kalman import predict_inplace, update_inplace


class HeartRateMethod(ImageToDataNode):
    """Heart Rate Determination Method"""

    def __init__(self, *args, **kwargs):
        # Initialise the "Node" object passing the name of our tracking method:
        super().__init__(*args, name="heart_rate", **kwargs)

        # Those headers specify the names of the quantities we will get out
        # of the tracking function. In our case ellipse position on x and y,
        # dimension on x and y, and orientation. With them, we generate the
        # "type" of the output, which will be a namedtuple with the specified
        #  keys.
        headers = ["avg_intensity", "heart_rate"]
        self._output_type = namedtuple("t", headers)

        # Monitored headers list specify which quantities will be displayed
        # in the streaming plot:
        self.monitored_headers = ["avg_intensity", "heart_rate"]

        # The data log name specify under which name we will find the tracked
        #  quantities in the log:
        self.data_log_name = "heart_rate"

        # To adjust the tracking, we can display diagnostic images instead of
        #  the raw image from the camera. Here we list the options:
        self.diagnostic_image_options = ["input", "test_diagnostic_image"]

        self.buffer = None
        self.framerate = 100
        self.temporal_window_s = 10

    def _process(
        self,
        im,
        wnd_pos: Param((100, 100), gui=False),
        wnd_dim: Param((100, 100), gui=False),
        framerate: Param(100.0, limits=(1, 200)),
        temporal_window_s: Param(10.0, limits=(1, None)),
        **extraparams,
    ):
        """
        :param im: input image
        :return: NodeOutput with the tracking results
        """
        # Diagnostic messages can be outputted with info on what went wrong:
        message = ""

        self.framerate = framerate
        v = temporal_window_s

        """
        TODO: Move cropping to dedicated method
        """
        cropped = im[
            wnd_pos[1] : wnd_pos[1] + wnd_dim[1],
            wnd_pos[0] : wnd_pos[0] + wnd_dim[0],
        ]

        # Calculate average intensity:
        avg_intensity = cropped.mean()
        heart_rate = 0

        # Calculate t_step
        dt = 1 / self.framerate

        if self.buffer is None:
            self._initialize_buffer()
        else:
            self.buffer[self.write_idx] = avg_intensity
            self.write_idx = (self.write_idx + 1) % self.N

            fft_input = self.buffer - np.mean(self.buffer)

            Y = np.fft.rfft(fft_input)
            freq = np.fft.rfftfreq(self.N, d=dt)
            amp = (2.0 / self.N) * np.abs(Y)
            freq_no_dc = freq[1:]
            amp_no_dc = amp[1:]
            idx_peak = np.argmax(amp_no_dc)
            heart_rate = freq_no_dc[idx_peak]
            # dominant_amp = amp_no_dc[idx_peak]
            # print(dominant_freq)

        if self.set_diagnostic == "input":
            # show the preprocessed, background-subtracted image
            self.diagnostic_image = im
        if self.set_diagnostic == "test_diagnostic_image":
            # show the thresholded image:
            self.diagnostic_image = im - im

        # Return a NodeOutput object which combines the message and the
        # output named tuple created from the output type defined in the init
        #  and the tuple with our tracked values
        return NodeOutput([message], self._output_type(avg_intensity, heart_rate))

    def _initialize_buffer(self):
        self.N = int(self.framerate * self.temporal_window_s)
        self.buffer = np.zeros(self.N, dtype=np.float32)
        self.write_idx = 0
