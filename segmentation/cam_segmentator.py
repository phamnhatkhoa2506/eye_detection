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
        
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame to segment eye components.
        
        Args:
            frame (np.ndarray): Input frame from camera
            
        Returns:
            tuple: (processed_frame, segmentation_info)
        """
        # Run YOLO segmentation
        results = self.model(frame, verbose=False)[0]
        
        # Create a copy of the frame for visualization
        processed_frame = frame.copy()
        
        # Initialize segmentation info
        segmentation_info = {
            'eye_area': 0,
            'iris': 0,
            'pupil': 0
        }
        
        # Process segmentation masks
        if hasattr(results, 'masks') and results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.data.cpu().numpy()
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                x1, y1, x2, y2, confidence, class_id = box
                class_name = self.classes[int(class_id)]
                
                # Update segmentation count
                segmentation_info[class_name] += 1
                
                # Convert mask to uint8
                mask = (mask * 255).astype(np.uint8)
                
                # Create colored mask
                colored_mask = np.zeros_like(processed_frame)
                colored_mask[mask > 0] = self.colors[class_name]
                
                # Blend mask with frame
                alpha = 0.5
                processed_frame = cv2.addWeighted(
                    processed_frame, 1,
                    colored_mask, alpha,
                    0
                )
                
                # Draw bounding box
                cv2.rectangle(
                    processed_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    self.colors[class_name],
                    2
                )
                
                # Add label with confidence
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(
                    processed_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.colors[class_name],
                    2
                )
        
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
