# Realtime Guide for Visually Impaired

## Overview

Realtime Guide for Visually Impaired is a project aimed at providing real-time assistance to visually impaired individuals by leveraging computer vision techniques. The project utilizes YOLOv4 (You Only Look Once version 4) for object detection, along with additional techniques to determine the object's location, coordinates, and distance from the camera. The system also features voice command functionality, allowing users to receive audible guidance about the object's position.

## Features

- Object Detection: YOLOv4 is employed to detect objects in the camera feed.
- Location Determination: The project determines the relative location of the detected object, indicating whether it is slightly left, slightly right, top, bottom, or in the middle of the camera frame.
- Coordinates Calculation: The coordinates of the object within the camera frame are determined, providing additional information about its position.
- Distance Estimation: By using a referential image and mathematical calculations, the system estimates the distance of the detected object from the camera.
- Voice Commands: Python libraries are utilized to implement voice command features, allowing users to receive verbal guidance about the object's location and distance.

## Getting Started

### Prerequisites

- Python 3.x
- YOLOv4 installed (refer to YOLOv4 documentation for installation instructions)
- Additional Python libraries (specified in requirements.txt)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/abhinav89908/Realtime-Guide-for-Visually-Impaired.git
    ```

2. Navigate to the project directory:

    ```bash
    cd realtime-guide-for-visually-impaired
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the main script:

    ```bash
    python app.py
    ```

2. Follow the voice prompts to receive real-time guidance about detected objects.

## Contributing

Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## Acknowledgments

- The YOLOv4 community for the powerful object detection model.
- Contributors to the Python libraries used in this project.

Feel free to reach out with any questions or feedback!
