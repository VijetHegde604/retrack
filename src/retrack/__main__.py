import argparse

import cv2

from retrack.detector import Detector
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

    args = parser.parse_args()

    if args.webcam:
        source = 0
    elif args.source:
        source = args.source
    else:
        parser.error("provide a video path or use --webcam")

    detector = Detector()
    video = VideoSource(source)

    for frame in video.frames():
        detections = detector.detect(frame)

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"{detection.class_id}: {detection.confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("ReTrack", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
