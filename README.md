# Car_Speed_Inspector
 This project analyzes road traffic from a video using Python and OpenCV. Cars are detected, tracked, and highlighted with color-coded rectangles based on their speed. Results, including detected speed and infractions, are saved in speedrecord.txt.
# Traffic Speed Radar (YOLOv10 + Homography + Tracking)

This project implements a virtual speed radar for vehicle detection and speed measurement in videos, using computer vision with YOLOv10 for detection, homography for perspective correction, and real-time velocity and acceleration estimation.

The system detects speeding violations, saves visual evidence, and generates automatic reports.

## Features

- 🚗 Vehicle detection and tracking (cars, motorcycles, heavy vehicles)
- 📏 Realistic speed calculation in km/h using homography transformation
- 📊 Acceleration estimation with smoothing (EMA) for stability
- 🎯 Soft-NMS to reduce false positives
- 🟥 Automatic detection of speeding violations with image snapshots
- 📄 Generation of TXT reports and data visualizations
- 🖥️ Export of 4K video with overlays for speed and acceleration

## Requirements

- Python 3.9+
- OpenCV
- Numpy
- Matplotlib
- Ultralytics YOLOv10 (nano)

## Installation

```bash
pip install -r requirements.txt
```
## Car_Speed_Inspector

.
├── resources/
│   └── traffic.mp4            # Input video
├── TrafficRecord/
│   ├── output_detection_4k.mp4 # Output video with detections
│   ├── exceeded/               # Images of speeding vehicles
│   ├── datavis.png             # Final visualization graph
│   └── SpeedRecord.txt         # Speed report
├── yolov10n.pt                # YOLOv10 nano pretrained model
├── main.py                    # Main source code
└── README.md                  # This documentation



