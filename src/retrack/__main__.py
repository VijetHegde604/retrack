import argparse

import cv2

from retrack.detector import Detector
from retrack.render import draw_detections
from retrack.video import VideoSource


def main() -> None:
    parser = argparse.ArgumentParser(description="ReTrack object detection")

    parser.add_argument(
        "source",
        nargs="?",
        help="Path to a video file",
    )

    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use the default webcam",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Opacity of coloured segmentation masks, from 0 to 1 (default: 0.45)",
    )

    args = parser.parse_args()

    if args.webcam:
        source = 0
    elif args.source:
        source = args.source
    else:
        parser.error("provide a video path or use --webcam")

    if not 0.0 <= args.mask_alpha <= 1.0:
        parser.error("--mask-alpha must be between 0 and 1")

    detector = Detector()
    video = VideoSource(source)

    for frame in video.frames():
        detections = detector.detect(frame)

        draw_detections(frame, detections, mask_alpha=args.mask_alpha)

        cv2.imshow("ReTrack", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
