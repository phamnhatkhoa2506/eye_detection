from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="./segmentation/Seg_Dataset_Eye/IRIS_PUPIL_EYE/data.yaml",  # đường dẫn đến file data.yaml
        epochs=10,
        batch=16,
        imgsz=640,                   # Resize input ảnh về 640x640
        task="segment",
        project="./segmentation",     # thư mục lưu kết quả
        name="yolov8n-seg-custom",   # tên experiment
        augment=True,                # bật sẵn augmentation
        degrees=10,                  # xoay ảnh ±10 độ
        translate=0.1,               # dịch ảnh 10%
        scale=0.5,                   # zoom ảnh 0.5x ~ 1.5x
        shear=5,                     # nghiêng hình ±5 độ
        perspective=0.0005,          # biến dạng phối cảnh nhẹ
        flipud=0.0,                  # không lật dọc (không phù hợp mắt)
        fliplr=0.5,                  # 50% ảnh lật ngang (giữ lại cho cân đối)
        hsv_h=0.015,                 # thay đổi hue
        hsv_s=0.7,                   # saturation
        hsv_v=0.4,                   # brightness
        mosaic=1.0,                  # bật mosaic (mix 4 ảnh)
        mixup=0.1,                   # nhẹ nhàng mixup
        copy_paste=0.1               # augmentation dán đối tượng
    )
