from collections.abc import Iterator

import cv2
import numpy as np


class VideoSource:
    """Read frames from a video file or webcam."""

    def __init__(self, source: str | int) -> None:
        self.source = source

    def frames(self) -> Iterator[np.ndarray]:
        capture = cv2.VideoCapture(self.source)

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                yield frame
        finally:
            capture.release()
