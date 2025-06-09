from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model
    
    # Train the model
    results = model.train(
        data='./Eye-Closed-Open/data.yaml',  # path to data config file
        epochs=10,                           # number of epochs
        imgsz=640,                          # image size
        batch=16,                           # batch size
        name='eye_detection'                # experiment name
    )