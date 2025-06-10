import os
import sys
import numpy as np
import cv2
import imutils
import torch
import supervision as sv
from typing import Optional
from ultralytics import YOLO
from datetime import datetime
from PIL import Image


# Configuration parameters
WIDTH = 800
start_status = "Analyzing..."
open_eye_status = "Open Eyes"
closed_eye_status = "Closed Eyes"

# Colors in BGR format
open_eye_color = (0, 255, 0)      # Green for open eyes
closed_eye_color = (0, 0, 255)    # Red for closed eyes
conf_color = (255, 255, 0)        # Yellow for confidence score
status_color = (255, 255, 255)    # White for status text
header_color = (255, 255, 255)    # White for header
footer_color = (255, 255, 255)    # White for footer

frame_name = "Eye State Detection"
quit_key = 'q'


class EyeDetector:
    """Class to handle eye state detection using YOLO model."""

    def __init__(
        self,
        weights_path: str,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(weights_path)
        self.cap = self._initialize_video_capture()
        self.frame_count = 0
        print(f"[INFO] Using device: {self.device}")

    def _load_model(self, weights_path: str) -> YOLO:
        try:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights not found at: {weights_path}")
            
            model = YOLO(weights_path).to(self.device)
            if not hasattr(model, 'predict'):
                raise AttributeError("Loaded model has no predict method")
            
            print(f"[INFO] Successfully loaded model from {weights_path}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            sys.exit(1)

    def _initialize_video_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Failed to open camera")
            sys.exit(1)
        return cap
        
    def _draw_detection(
        self,
        frame: np.ndarray,
        box_data: np.ndarray
    ) -> str:
        xywh = np.array(box_data.boxes.xywh.cpu()).astype("int32")
        xyxy = np.array(box_data.boxes.xyxy.cpu()).astype("int32")
        cc_data = np.array(box_data.boxes.data.cpu())
        status = start_status
        
        for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, conf, class_) in zip(xyxy, xywh, cc_data):
            if class_ == 1:  # Open eyes
                self._draw_eye_box(frame, x1, y1, w, h, conf, open_eye_color, "Open")
                status = open_eye_status
            elif class_ == 0:  # Closed eyes
                self._draw_eye_box(frame, x1, y1, w, h, conf, closed_eye_color, "Closed")
                status = closed_eye_status

        return status
    
    def _draw_eye_box(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        w: int,
        h: int,
        conf: float,
        color: tuple,
        state: str
    ) -> None:
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        
        # Draw state label
        label = f"{state}: {np.round(conf * 100, 1)}%"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_header_footer(self, frame: np.ndarray, status: str) -> None:
        # Header
        header_text = f"Eye State Detection - {datetime.now().strftime('%H:%M:%S')}"
        cv2.putText(frame, header_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2)
        
        # Footer
        footer_text = f"Status: {status}"
        frame_height = frame.shape[0]
        cv2.putText(frame, footer_text, (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, footer_color, 2)

    def process_video(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] Failed to capture frame")
                break

            self.frame_count += 1
            frame = imutils.resize(frame, width=WIDTH)

            try:
                result = self.model.predict(frame)
                if result is None or not result:
                    print(f"[WARNING] No detection in frame {self.frame_count}")
                    continue
            except Exception as e:
                print(f"[ERROR] Prediction failed: {str(e)}")
                break

            status = start_status
            if len(result[0].boxes) > 0:
                status = self._draw_detection(frame, result[0])
            
            self._draw_header_footer(frame, status)
            cv2.imshow(frame_name, frame)

            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                print("[INFO] Quit key pressed")
                break

    def cleanup(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Program terminated successfully")


def main():    
    detector = EyeDetector(
        weights_path="./best2.pt",
    )
    try:
        detector.process_video()
    finally:
        detector.cleanup()


if __name__ == '__main__':
    main()