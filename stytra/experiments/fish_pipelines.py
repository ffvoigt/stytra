from stytra.gui.camera_display import (
    CameraViewFish,
    EyeTailTrackingSelection,
    EyeTrackingSelection,
    HeartRateSelection,
    TailTrackingSelection,
)
from stytra.gui.fishplots import BoutPlot, TailStreamPlot
from stytra.tracking.eyes import EyeTrackingMethod
from stytra.tracking.fish import FishTrackingMethod
from stytra.tracking.heart import HeartRateMethod
from stytra.tracking.pipelines import Pipeline
from stytra.tracking.preprocessing import BackgroundSubtractor, Prefilter
from stytra.tracking.tail import CentroidTrackingMethod


class TailTrackingPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.filter = Prefilter(parent=self.root)
        self.tailtrack = CentroidTrackingMethod(parent=self.filter)
        self.extra_widget = TailStreamPlot
        self.display_overlay = TailTrackingSelection


class TailTrackingPipeline2(Pipeline):
    def __init__(self):
        super().__init__()
        self.filter = Prefilter(parent=self.root)
        self.tailtrack = CentroidTrackingMethod(parent=self.filter)
        self.extra_widget = TailStreamPlot
        self.display_overlay = TailTrackingSelection


class FishTrackingPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.bgsub = BackgroundSubtractor(parent=self.root)
        self.fishtrack = FishTrackingMethod(parent=self.bgsub)
        self.extra_widget = BoutPlot
        self.display_overlay = CameraViewFish


class EyeTrackingPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        # self.filter = Prefilter(parent=self.root)
        self.eyetrack = EyeTrackingMethod(parent=self.root)
        self.display_overlay = EyeTrackingSelection


class EyeTailTrackingPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.filter = Prefilter(parent=self.root)
        self.tailtrack = CentroidTrackingMethod(parent=self.filter)

        self.eyetrack = EyeTrackingMethod(parent=self.root)
        self.display_overlay = EyeTailTrackingSelection


class HeartRatePipeline(Pipeline):
    def __init__(self):
        super().__init__()
        # self.heartcrop = CropImage(parent=self.root)
        self.hearttrack = HeartRateMethod(parent=self.root)
        self.display_overlay = HeartRateSelection


pipeline_dict = dict(
    tail=TailTrackingPipeline,
    fish=FishTrackingPipeline,
    eyes=EyeTrackingPipeline,
    eyes_tail=EyeTailTrackingPipeline,
    heart=HeartRatePipeline,
)
