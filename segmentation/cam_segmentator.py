import cv2
import numpy as np
from ultralytics import YOLO
import time


class EyeSegmentation:
    def __init__(self, model_path: str = './segmentation/best.pt') -> None:
        """
        Initialize the eye segmentation model.
        
        Args:
            model_path (str): Path to the trained YOLO segmentation model weights
        """
        self.model = YOLO(model_path)
        self.classes = ['eye_area', 'iris', 'pupil']
        self.colors = {
            'eye_area': (255, 0, 0),    # Blue for eye area
            'iris': (0, 255, 0),        # Green for iris
            'pupil': (0, 0, 255)        # Red for pupil
        }

    def _detect_eyes(self, gray_img: np.ndarray, scale_factor: float = 10, min_neighbors: int = 3) -> list:
        """
        Detect eyes in the grayscale image using Haar cascade.
        
        Args:
            gray_img (np.ndarray): Grayscale input image
            scale_factor (float): Scale factor for detection
            min_neighbors (int): Minimum neighbors for detection
            
        Returns:
            list: List of eye regions (x, y, w, h)
        """
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        return eyes

    def _process_eye_region(self, img: np.ndarray, eye_region: tuple) -> np.ndarray:
        """
        Process a single eye region using the segmentation model.
        
        Args:
            img (np.ndarray): Original image
            eye_region (tuple): Eye region coordinates (x, y, w, h)
            
        Returns:
            np.ndarray: Processed eye region
        """
        x, y, w, h = eye_region
        
        # Crop and resize eye region
        eye_crop = img[y:y+h, x:x+w]
        eye_crop_resized = cv2.resize(eye_crop, (640, 640))
        eye_gray = cv2.cvtColor(eye_crop_resized, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.merge([eye_gray]*3)
        
        # Get segmentation mask
        result = self.model.predict(gray_img, verbose=False)[0]
        
        if hasattr(result, 'masks') and result.masks is not None:
            mask_array = result.masks.data[0].cpu().numpy()
            
            # Resize mask to original eye region size
            mask_resized = cv2.resize(mask_array.astype(np.uint8) * 255, (w, h))
            
            # Create overlay
            overlay = eye_crop.copy()
            overlay[mask_resized > 127] = (0, 255, 0)  # Green color
            
            # Blend overlay with original
            blended = cv2.addWeighted(eye_crop, 0.7, overlay, 0.3, 0)
            return blended
        
        return eye_crop

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame to segment eye components.
        
        Args:
            frame (np.ndarray): Input frame from camera
            
        Returns:
            tuple: (processed_frame, segmentation_info)
        """
        # Convert frame to grayscale for eye detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a copy of the frame for visualization
        processed_frame = frame.copy()
        
        # Initialize segmentation info
        segmentation_info = {
            'eye_area': 0,
            'iris': 0,
            'pupil': 0
        }
        
        # Detect eyes
        eyes = self._detect_eyes(gray)
        
        # Process each eye region
        for eye_region in eyes:
            processed_region = self._process_eye_region(processed_frame, eye_region)
            x, y, w, h = eye_region
            processed_frame[y:y+h, x:x+w] = processed_region
            
            # Update segmentation info
            segmentation_info['eye_area'] += 1
        
        return processed_frame, segmentation_info
    
    def process_camera(self, camera_id: int = 0) -> None:
        """
        Process camera feed in real-time.
        
        Args:
            camera_id (int): Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame, segmentation_info = self.process_frame(frame)
            
            # Add segmentation info to frame
            y_offset = 30
            for component, count in segmentation_info.items():
                text = f"{component}: {count}"
                cv2.putText(
                    processed_frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.colors[component],
                    2
                )
                y_offset += 30
            
            # Display frame
            cv2.imshow('Eye Segmentation', processed_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    # Initialize segmentator
    segmentator = EyeSegmentation()
    
    # Start camera processing
    segmentator.process_camera()


if __name__ == "__main__":
    main()
