# Real-Life Gesture Control

## Overview

This project implements a real-time finger counting and gesture recognition system using OpenCV and MediaPipe. The software can detect hand gestures via a webcam and trigger corresponding actions based on the number of fingers extended.

## Features

- **Real-time hand tracking** using MediaPipe
- **Finger counting** for both left and right hands
- **Gesture-based actions**:
  - 0 fingers: "Turning OFF lights"
  - 1 finger: "Activating device"
  - 2 fingers: "Scrolling UP"
  - 3 fingers: "Scrolling DOWN"
  - 4 fingers: "Adjusting volume"
  - 5 fingers: "Turning ON lights"
- **Debounce mechanism** to ensure stable gesture recognition
- **FPS display** for performance monitoring
- **Screenshot capture** by pressing 's'
- **Exit the program** by pressing 'q'

## Installation

### Prerequisites

Make sure you have Python installed (Python 3.6+ is recommended). Install the required dependencies:

```sh
pip install opencv-python mediapipe
```

## Usage

Run the script using:

```sh
python main.py
```

### Controls

- **Press 'q'** to quit the application.
- **Press 's'** to save a screenshot.

## How It Works

1. The program captures frames from the webcam.
2. It processes the frame using MediaPipe to detect hands and finger positions.
3. The number of extended fingers is counted and mapped to predefined actions.
4. If the same gesture is detected for a set number of frames (debounce threshold), the corresponding action is triggered.
5. The FPS (frames per second) is displayed for performance monitoring.

## File Structure

```
.
├── main.py  # Main script
├── README.md          # Project documentation
├── requirements.txt   # List of dependencies (optional)
```

## Dependencies

- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- Python Standard Libraries (`time`)

## Potential Enhancements

- Implement integration with IoT devices for real-world automation.
- Add support for more complex gestures.
- Train a custom model for specific hand gestures beyond finger counting.

## License

This project is open-source under the MIT License.

## Author

Developed by [Yong Hsiang Hsieh]

