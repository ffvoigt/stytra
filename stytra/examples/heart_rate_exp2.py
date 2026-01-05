from pathlib import Path
from stytra import Stytra
from stytra.stimulation import Protocol

REQUIRES_EXTERNAL_HARDWARE = False


class HeartRateProtocol(Protocol):
    name = "heart_rate_measurement"

    # To add tracking to a protocol, we simply need to add a tracking
    # argument to the stytra_config:
    stytra_config = dict(
        tracking=dict(embedded=True, method="heart"),
        camera=dict(
            video_file=str(Path(__file__).parent / "assets" / "fish_compressed.h5")
        ),
    )


if __name__ == "__main__":
    s = Stytra(protocol=HeartRateProtocol())
