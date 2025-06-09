import os
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


label_to_index = {
    'closed_eyes': 0,
    'open_eyes': 1
}

WIDTH, HEIGHT = 512, 512


def load_data(
    data_name: str="MichalMlodawski/closed-open-eyes", 
    num_samples: int=2000
) -> List[Dict[str, Any]]:
    ds = load_dataset(data_name)
    ds_shuffled = ds["train"].shuffle(seed=42)
    rows = [row for row in ds_shuffled.select(range(num_samples))]

    return rows


def split_data(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]]]:
    train_ds, val_ds = train_test_split(rows, test_size=0.2, random_state=42)
    val_ds, test_ds = train_test_split(val_ds, test_size=0.2, random_state=42)

    return train_ds, val_ds, test_ds


def normalize_to_YOLO_data(x_center: float, y_center: float, w: float, h: float) -> list[float]:
    normalized_x = x_center / WIDTH
    normalized_y = y_center / HEIGHT
    normalized_w = w / WIDTH
    normalized_h = h / HEIGHT

    return normalized_x, normalized_y, normalized_w, normalized_h


def make_YOLO_data(dir_path: str, ds: list[dict]) -> None:
    img_dir_path = os.path.join(dir_path, "images")
    label_dir_path = os.path.join(dir_path, "labels")
    
    os.makedirs(img_dir_path, exist_ok=True)
    os.makedirs(label_dir_path, exist_ok=True)

    for i, item in tqdm(enumerate(ds)):
        # Get eye info
        le = normalize_to_YOLO_data(*item["Left_eye_react"])
        re = normalize_to_YOLO_data(*item["Right_eye_react"])

        # Get label Index
        index = label_to_index[item["Label"]]

        # Get image data
        img_data = item["Image_data"]
        img_pil: Image = img_data["file"]
        filename = img_data["filename"]
        
        img_pil.save(os.path.join(img_dir_path, filename + ".png"))

        with open(os.path.join(label_dir_path, filename + '.txt'), "w")  as f:
            f.write(f"{index} {le[0]} {le[1]} {le[2]} {le[3]}\n{index} {re[0]} {re[1]} {re[2]} {re[3]}")
    

def main() -> None:
    rows = load_data()
    train_ds, val_ds, test_ds = split_data(rows)

    base_dir = "./Eye-Closed-Open"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    val_dir = os.path.join(base_dir, "valid")

    make_YOLO_data(train_dir, train_ds)
    make_YOLO_data(val_dir, val_ds)
    make_YOLO_data(test_dir, test_ds)


if __name__ == "__main__":
    main()