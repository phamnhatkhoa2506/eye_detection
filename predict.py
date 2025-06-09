import supervision as sv
import cv2
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
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        Image.fromarray(annotated_image).save(output_path)


if __name__ == "__main__":
    weights_path = ''
    img_paths = [

    ]
    output_path = './images_predictions'

    detect_image(weights_path, img_paths, output_path)