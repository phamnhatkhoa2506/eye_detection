# Eye State Detection (Open/Closed) with YOLO

This project provides a real-time eye state (open/closed) detection system using a YOLO-based deep learning model. It includes scripts for live webcam detection, image-based detection, and dataset preparation/training.

## Features

- **Real-time webcam detection**: Detects whether eyes are open or closed using your webcam.
- **Image detection**: Run detection on static images.
- **Custom training**: Tools to prepare and train your own dataset.
- **YOLOv8-based**: Utilizes the latest YOLO models for fast and accurate detection.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Getting Started

### 1. Clone the repository

```sh
git clone <repo-url>
cd eye_detection
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Download or train a model

- The default webcam detector uses `best.pt` (YOLO weights). Place your trained weights as `best.pt` in the project root, or use the provided one.

### 4. Run the webcam detector

```sh
python cam_detector.py
```

- Press `q` to quit the webcam window.

### 5. Run detection on images

Edit `image_detector.py` to set your image paths and model weights, then run:

```sh
python image_detector.py
```

### 6. Training

To train your own model, use:

```sh
python train.py
```

Edit `train.py` to adjust training parameters and dataset paths as needed.

### 7. Dataset Preparation

Use `data.py` to download and prepare datasets in YOLO format.

```sh
python data.py
```

## Dataset Format

The dataset should be structured as follows (see `dataset/data.yaml`):

```
dataset/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
```

Classes:
- `0`: closed
- `1`: open

## Docker Usage

To run the webcam detector in Docker:

```sh
docker build -t eye-cam-detector .
docker run --rm --device=/dev/video0 eye-cam-detector
```

> **Note:** You may need to adjust the `--device` flag for webcam access depending on your OS.

## Files Overview

- `cam_detector.py` — Real-time webcam eye state detection.
- `image_detector.py` — Detects eye state in images.
- `data.py` — Dataset download and YOLO-format preparation.
- `train.py` — Model training script.
- `requirements.txt` — Python dependencies.
- `best.pt` — YOLO model weights (not included, add your own or train).
