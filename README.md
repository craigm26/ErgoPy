# ErgoPy

Desktop posture monitor built with Tkinter, OpenCV, and MediaPipe. ErgoPy watches your webcam feed, tracks key body landmarks, and gives real-time feedback (visual HUD + Windows alert chime) when it detects slouching, leaning, or a dropped head posture.

## Features

- Live webcam feed with overlayed pose skeleton and baseline “ghost” line.
- One-click calibration to capture your neutral posture.
- Posture health meter that recovers when you correct your pose and depletes when you slip.
- Alerts (beep + red status) when slouching, leaning, or looking down.
- Simple Tkinter UI sized for a desktop side panel, with adjustable sensitivity sliders and camera index reconnect.
- On-screen setup instructions, camera reconnect, FPS meter, and console logs for camera/posture events to verify everything is working.

## Prerequisites

- Python 3.10+ (Tkinter included on most installations).
- Webcam accessible at index 0.
- Windows for the current alert sound (`winsound`). Other platforms can run the UI but won’t play the built-in beep.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python mediapipe pillow numpy
```

## Run

```bash
python ErgoPy.py
```

## How to Use

1. Launch the app; the canvas will show your mirrored webcam feed with a basic skeleton.
2. Sit in your ideal, neutral posture and click **SET BASELINE** to calibrate.
3. Keep the window visible; status text and the health meter update in real time.
4. If posture drifts, the status turns red, health decreases, and a Windows alert chimes (debounced to avoid spam). Click **RE-CALIBRATE** anytime you want a new baseline. **RESET** clears calibration and health.

## What It Detects

- **Slouching:** Shoulders drop vs. baseline.
- **Leaning:** Shoulder center shifts sideways vs. baseline.
- **Head Down (tech neck):** Nose drops relative to baseline.

Thresholds are tuned for a typical seated position and can be tweaked in `check_posture` inside `ErgoPy.py` if you need stricter or looser sensitivity.

## Troubleshooting

- **“ERROR: CAMERA NOT FOUND”** – Ensure another app isn’t using the webcam; verify the camera index (default 0) in `cv2.VideoCapture(0)`.
- **No pose detected** – Check lighting and ensure your upper body is visible; avoid very dark or backlit scenes.
- **No audio alert on non-Windows OS** – `winsound` is Windows-only; swap in another audio cue for your platform if needed.

## Notes

- All UI styling lives inline in `ErgoPy.py`; there is no external config file.
- The included GitHub workflow (`.github/workflows/python-publish.yml`) is a template for building and publishing Python distributions on release.
