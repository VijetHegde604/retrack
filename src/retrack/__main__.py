import argparse
import cv2

from retrack.detector import Detector
from retrack.render import draw_detections
from retrack.video import VideoSource
from retrack.bytetracker import ByteTracker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReTrack object detection and tracking"
    )

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
        help="Opacity of coloured segmentation masks",
    )

    args = parser.parse_args()

    # Select video source
    if args.webcam:
        source = 0
    elif args.source:
        source = args.source
    else:
        parser.error("provide a video path or use --webcam")

    # Check mask opacity
    if not 0.0 <= args.mask_alpha <= 1.0:
        parser.error("--mask-alpha must be between 0 and 1")

    # Create detector
    detector = Detector()

    # Create tracker
    tracker = ByteTracker(
    high_threshold=0.5,
    low_threshold=0.1,
    iou_threshold=0.3,
    max_lost=30,
)

    # Create video source
    video = VideoSource(source)

    # Process video frame by frame
    for frame in video.frames():

        # 1. Detect objects
        detections = detector.detect(frame)

        # 2. Assign tracking IDs
        tracks = tracker.update(detections)

        # Debug information in terminal
        print(
            "DETECTIONS:",
            len(detections),
            "TRACKS:",
            len(tracks),
        )

        for track in tracks:
            print(
                track.class_name,
                "ID:",
                track.track_id,
            )

        # 3. Draw YOLO detections
        draw_detections(
            frame,
            detections,
            mask_alpha=args.mask_alpha,
        )

        # 4. Draw tracking IDs
        for track in tracks:

            x1, y1, x2, y2 = map(
                int,
                track.bbox,
            )

            label = (
                f"{track.class_name} "
                f"ID:{track.track_id}"
            )

            # Put the label on the right side
            # of the bounding box
            cv2.putText(
                frame,
                label,
                (x2 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # 5. Show frame
        cv2.imshow(
            "ReTrack",
            frame,
        )

        # 6. Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()