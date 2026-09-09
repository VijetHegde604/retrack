"""Main CLI entry point for ReTrack."""

from __future__ import annotations

import argparse
import time
import cv2

from retrack.detector import Detector
from retrack.byte_tracker import ByteTrackTracker
from retrack.render import draw_tracks
from retrack.video import VideoSource


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReTrack - CPU-efficient multi-object tracking with persistent ReID"
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
        "--model",
        type=str,
        default="yolo11s-seg.pt",
        help="YOLO model path or identifier (e.g. yolo11s-seg.pt or yolo11n.pt)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: 'auto', 'cpu', 'cuda', or 'mps'",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (e.g. 640 for speed)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold",
    )

    parser.add_argument(
        "--reid-thresh",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for ReID object re-identification",
    )

    parser.add_argument(
        "--lost-buffer",
        type=int,
        default=30,
        help="Frames before an unobserved track is moved from short-term to persistent gallery",
    )

    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Disable ReID appearance matching (revert to pure motion ByteTrack)",
    )

    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Opacity of coloured segmentation masks (0.0 for outlines only)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the annotated video file",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without opening an OpenCV window",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process",
    )

    args = parser.parse_args()

    # Select video source
    if args.webcam:
        source = 0
    elif args.source:
        source = args.source
    else:
        parser.error("provide a video path or use --webcam")

    if not 0.0 <= args.mask_alpha <= 1.0:
        parser.error("--mask-alpha must be between 0 and 1")

    # Initialize detector
    detector = Detector(
        model=args.model,
        confidence=args.conf,
        device=args.device,
        imgsz=args.imgsz,
    )

    # Initialize persistent tracker
    tracker = ByteTrackTracker(
        track_activation_threshold=args.conf,
        high_conf_threshold=max(args.conf, 0.45),
        low_conf_threshold=0.1,
        lost_track_buffer=args.lost_buffer,
        reid_threshold=args.reid_thresh,
        reid_device=args.device,
        enable_reid=not args.no_reid,
    )

    video = VideoSource(source)

    # Video writer setup if requested
    writer = None
    if args.output:
        # Probe video properties
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_count = 0
    start_time = time.time()

    print("Starting ReTrack...")
    print(f"Device: {detector.device} | Model: {args.model} | ReID Enabled: {not args.no_reid}")

    try:
        for frame in video.frames():
            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                break

            t0 = time.time()

            # 1. Detect objects
            detections = detector.detect(frame)

            # 2. Update persistent tracker (with frame for selective ReID)
            tracks = tracker.update(detections, frame=frame)

            dt = time.time() - t0
            fps_instant = 1.0 / max(1e-4, dt)

            # 3. Class names dictionary
            class_names = {
                d.class_id: d.class_name
                for d in detections
            }

            # 4. Render tracks with persistent IDs & badges
            draw_tracks(
                frame,
                tracks,
                class_names=class_names,
                mask_alpha=args.mask_alpha,
            )

            # 5. Terminal status
            active_info = [
                f"{class_names.get(t.class_id, 'obj')}#{t.track_id}{' (RETRACK)' if t.reidentified else ''}"
                for t in tracks
            ]
            print(
                f"[Frame {frame_count:04d}] {fps_instant:4.1f} FPS | "
                f"Dets: {len(detections):2d} | Active Tracks: {len(tracks):2d} | {', '.join(active_info)}"
            )

            # 6. Save or display
            if writer:
                writer.write(frame)

            if not args.no_display:
                # Optionally downscale display if 4K
                disp_frame = frame
                if frame.shape[1] > 1920:
                    disp_frame = cv2.resize(frame, (1920, int(frame.shape[0] * 1920 / frame.shape[1])))

                cv2.imshow("ReTrack", disp_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        if writer:
            writer.release()
            print(f"Saved annotated video to {args.output}")
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {total_time:.2f}s ({frame_count / max(1e-4, total_time):.1f} avg FPS)")


if __name__ == "__main__":
    main()
