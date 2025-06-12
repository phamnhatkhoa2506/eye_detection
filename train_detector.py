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
        name='eye_detection',               # experiment name
        
        # Data augmentation parameters
        augment=True,                       # enable augmentation
        degrees=10.0,                       # rotation augmentation (+/- deg)
        translate=0.1,                      # image translation (+/- fraction)
        scale=0.5,                          # image scale (+/- gain)
        shear=2.0,                          # image shear (+/- deg)
        perspective=0.0,                    # image perspective (+/- fraction)
        flipud=0.0,                         # image flip up-down (probability)
        fliplr=0.5,                         # image flip left-right (probability)
        mosaic=1.0,                         # image mosaic (probability)
        mixup=0.0,                          # image mixup (probability)
        copy_paste=0.0,                     # segment copy-paste (probability)
        
        # Scaling parameters
        hsv_h=0.015,                       # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,                         # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,                         # image HSV-Value augmentation (fraction)
        auto_augment='randaugment',        # auto augmentation policy
        rect=False,                         # rectangular training
        cos_lr=True,                        # cosine learning rate scheduler
        close_mosaic=10,                   # disable mosaic augmentation for final epochs
    )