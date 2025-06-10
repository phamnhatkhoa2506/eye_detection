import supervision as sv
import cv2
import os
from ultralytics import YOLO
from PIL import Image


def detect_image(weights_path: str, img_paths: list[str], output_path: str) -> None:
    model = YOLO(weights_path)

    for img_path in img_paths:
        image = cv2.imread(img_path)

        results = model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        # annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)  

        output_filename = f"{output_path}/{os.path.basename(img_path)}"
        Image.fromarray(annotated_image).save(output_filename)


if __name__ == "__main__":

    weights_path = './best2.pt'
    img_paths = [
        './Eye-Closed-Open/test/images/0c145ba43b61fc25f010dac19a9664665abe9d0abbe638eea8d1cec1f050aabc.jpeg.png',
        './Eye-Closed-Open/test/images/1a57623029c3a4a97945b76fef30e0cc35f8d132eb4366b6858a57c3428c20a1.jpeg.png',
        './Eye-Closed-Open/test/images/2b8afb65db23eda3eda02969008347e2b0122c75229bbe73484a157079d34294.jpeg.png'
    ]
    output_path = './images_predictions'

    detect_image(weights_path, img_paths, output_path)