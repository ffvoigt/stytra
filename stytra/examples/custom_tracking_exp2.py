from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from lightparam import Param

from stytra import Stytra
from stytra.gui.camera_display import CameraSelection
from stytra.stimulation import Protocol
from stytra.stimulation.stimuli import Pause
from stytra.tracking.pipelines import (
    ImageToDataNode,
    ImageToImageNode,
    NodeOutput,
    Pipeline,
)
from stytra.tracking.preprocessing import BackgroundSubtractor

REQUIRES_EXTERNAL_HARDWARE = False

# Here we showcase the steps required to add your custom tracking function in
#  stytra. It might look a bit complicated at the beginning, but
# following closely the example and use this script as a template will make
# it easier.


# First of all, we need to create a new tracking method. You should have read
#  in the documentation that in Stytra the image processing analysis is
# defined in a branching way, appending nodes for image transformations (
# ImageToImage nodes) followed by nodes for extracting numbers, such as
# animal position or posture (ImageToDataNode). For this particular example,
# we will use the powerful adaptive background subraction that we just import
#  from stytra in line 2, followed by the following new custo method.

# To track the drosophila, we will fit an ellipse to its body, and we will
# log the 5 variables of the ellipse (xy position, yx dimension, orientation).


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
        headers = ["avg_intensity"]
        self._output_type = namedtuple("t", headers)

        # Monitored headers list specify which quantities will be displayed
        # in the streaming plot:
        self.monitored_headers = ["avg_intensity"]

        # The data log name specify under which name we will find the tracked
        #  quantities in the log:
        self.data_log_name = "heart_avg_intensity"

        # To adjust the tracking, we can display diagnostic images instead of
        #  the raw image from the camera. Here we list the options:
        self.diagnostic_image_options = ["input", "test_diagnostic_image"]

        self.previous_data = None

    def _process(
        self,
        im,
        wnd_pos: Param((129, 20), gui=False),
        wnd_dim: Param((14, 22), gui=False),
        **extraparams,
    ):
        """
        :param im: input image
        :return: NodeOutput with the tracking results
        """
        # Diagnostic messages can be outputted with info on what went wrong:
        message = ""

        cropped = im[
            wnd_pos[1] : wnd_pos[1] + wnd_dim[1],
            wnd_pos[0] : wnd_pos[0] + wnd_dim[0],
        ]

        # Calculate average intensity:
        avg_intensity = cropped.mean()

        if self.set_diagnostic == "input":
            # show the preprocessed, background-subtracted image
            self.diagnostic_image = im
        if self.set_diagnostic == "test_diagnostic_image":
            # show the thresholded image:
            self.diagnostic_image = im - im

        # Return a NodeOutput object which combines the message and the
        # output named tuple created from the output type defined in the init
        #  and the tuple with our tracked values
        return NodeOutput([message], self._output_type((avg_intensity)))


class HeartRateSelection(CameraSelection):
    def __init__(self, **kwargs):
        """ """
        super().__init__(**kwargs)

        # We need to initialise the rectangular ROI, add it to the area, and remove
        # the handles from the ellipseROI:
        self.heart_params = self.experiment.pipeline.hearttrack._params
        self.roi_pen = dict(color=(40, 5, 200), width=3)
        self.heart_roi = pg.RectROI(
            pos=(0, 0), size=(10, 10), movable=True, pen=self.roi_pen
        )
        self.initialise_roi(self.heart_roi)

    def retrieve_image(self):
        """
        This is the function that is called from the Stytra GUI at every
        update loop.
        Note that this function run in the same process of the rest of the
        GUI and of the stimulus. If you put here some slow code, it will slow
        down the entire interface and the stimulation update as well!
        """
        super().retrieve_image()

        # Pass if there is still no image from the camera:
        if self.current_image is None:
            return

    def set_pos_from_tree(self):
        """Go to parent for definition."""
        super().set_pos_from_tree()
        if not self.setting_param_val:
            self.heart_roi.setPos(self.heart_params.wnd_pos, finish=False)
            self.heart_roi.setSize(self.heart_params.wnd_dim)

    def set_pos_from_roi(self):
        """Go to parent for definition."""
        super().set_pos_from_roi()

        self.setting_param_val = True
        self.heart_params.params.wnd_dim.changed = True
        self.heart_params.wnd_dim = tuple([int(p) for p in self.heart_roi.size()])
        self.heart_params.params.wnd_pos.changed = True
        self.heart_params.wnd_pos = tuple([int(p) for p in self.heart_roi.pos()])
        self.setting_param_val = False


# Finally, we assemble a "pipeline" where we specify the order of the
# functions that we will apply to the image. Each pipeline node will take as
# "parent" the node from which it reads the input frames.
# The "display overlay" attribute, which does not necessarily have to be
# specified, will allow us to overimpose on the GUI an ROI to monitor
# live our tracking.


class HeartPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        # self.heartcrop = CropImage(parent=self.root)
        self.hearttrack = HeartRateMethod(parent=self.root)
        self.display_overlay = HeartRateSelection


class HeartRateProtocol(Protocol):
    name = "heart_rate_display"
    stytra_config = dict(
        tracking=dict(method=HeartPipeline),
        camera=dict(
            video_file=str(Path(__file__).parent / "assets" / "fish_compressed.h5")
        ),
    )

    def get_stim_sequence(self):
        # Empty protocol of specified duration:
        return [Pause(duration=10)]


if __name__ == "__main__":
    s = Stytra(protocol=HeartRateProtocol())
