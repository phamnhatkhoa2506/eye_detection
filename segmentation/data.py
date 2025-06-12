import os
import numpy as np
import cv2
from tqdm import tqdm

# Định nghĩa kích thước ảnh chuẩn hóa
WIDTH = 480
HEIGHT = 640

def make_YOLO_data(dir_path: str, task_dir: str):
    img_dir = f"{dir_path}/{task_dir}/images"
    mask_dir = f"{dir_path}/{task_dir}/segmentation"

    out_label_dir = f"{dir_path}/{task_dir}/labels"

    os.makedirs(out_label_dir, exist_ok=True)

    for file in tqdm(os.listdir(img_dir)):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Đọc ảnh mask (giả sử là ảnh đơn kênh, mỗi pixel là class_id)
        mask_path = os.path.join(mask_dir, file)
        mask = cv2.imread(mask_path, 0)

        h, w = mask.shape

        label_path = os.path.join(out_label_dir, file.replace(".png", ".txt").replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for class_id in np.unique(mask):
                if class_id == 0:  # bỏ qua vùng không quan tâm nếu dùng 255 làm ignore label
                    continue

                binary_mask = (mask == class_id).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) < 3:
                        continue  # bỏ qua các polygon quá nhỏ

                    # Chuẩn hóa polygon điểm
                    contour = contour.reshape(-1, 2)
                    segmentation = ""
                    for (px, py) in contour:
                        segmentation += f"{px / w:.6f} {py / h:.6f} "

                    # Ghi dòng dữ liệu
                    yolo_line = f"{class_id - 1} {segmentation.strip()}\n"
                    f.write(yolo_line)


if __name__ == "__main__":
    make_YOLO_data('./segmentation/Seg_Dataset_Eye/IRIS_PUPIL_EYE', "train")
    make_YOLO_data('./segmentation/Seg_Dataset_Eye/IRIS_PUPIL_EYE', "val")