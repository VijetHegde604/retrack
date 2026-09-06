# retrack
CPU-Efficient Long-Term Multi-Object Tracking with Selective Re-Identification

## Segmentation visualization

The default YOLO11 segmentation model renders each detected mask with a
semi-transparent, per-instance colour and a matching outline. Bounding boxes
and labels use the same colour. Use `--mask-alpha` to control the fill opacity:

```bash
uv run retrack path/to/video.mp4 --mask-alpha 0.55
```

Set `--mask-alpha 0` to show outlines without a coloured fill. Press `q` to
close the video window.
