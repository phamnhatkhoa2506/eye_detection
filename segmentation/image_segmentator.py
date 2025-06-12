import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def load_image(image_path: str) -> tuple:
    """
    Load and preprocess the input image.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        tuple: (original image, grayscale image)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def detect_eyes(gray_img: np.ndarray, scale_factor: float = 10, min_neighbors: int = 3) -> list:
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


def process_eye_region(img: np.ndarray, eye_region: tuple, model: YOLO) -> np.ndarray:
    """
    Process a single eye region using the segmentation model.
    
    Args:
        img (np.ndarray): Original image
        eye_region (tuple): Eye region coordinates (x, y, w, h)
        model (YOLO): Segmentation model
        
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
    result = model.predict(gray_img, save=True, name=f"img_{x}_{y}", project="./runs2")
    mask_array = result[0].masks.data[0].cpu().numpy()
    
    # Resize mask to original eye region size
    mask_resized = cv2.resize(mask_array.astype(np.uint8) * 255, (w, h))
    
    # Create overlay
    overlay = eye_crop.copy()
    overlay[mask_resized > 127] = (0, 255, 0)  # Green color
    
    # Blend overlay with original
    blended = cv2.addWeighted(eye_crop, 0.7, overlay, 0.3, 0)
    
    return blended


def process_image(image_path: str, model_path: str = 'runs/segment/eye_segmentation/weights/best.pt') -> None:
    """
    Process the entire image for eye segmentation.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the segmentation model
    """
    # Load model
    model = YOLO(model_path)
    
    # Load and preprocess image
    img, gray = load_image(image_path)
    
    # Detect eyes
    eyes = detect_eyes(gray)
    
    # Process each eye region
    for eye_region in eyes:
        processed_region = process_eye_region(img, eye_region, model)
        x, y, w, h = eye_region
        img[y:y+h, x:x+w] = processed_region
    
    # Save result
    cv2.imwrite("./seg_result.jpg", img)
    print("Segmentation completed. Result saved as 'seg_result.jpg'")


def main() -> None:
    """
    Main function to run the eye segmentation process.
    """
    try:
        process_image("./test_face.jpg")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
