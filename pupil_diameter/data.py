import os
import shutil
from zipfile import ZipFile, Path as ZipPath

count_files = 0

def copy_files(src_dir: ZipPath, dst_dir: str, num_subdirs: int):

    global count_files

    count_subdirs = 0
    for subdir in src_dir.iterdir():
        if not subdir.is_dir():
            continue
        if count_subdirs >= num_subdirs:
            break
        count_subdirs += 1

        for file in subdir.iterdir():
            if not file.name.lower().endswith(('.png', '.npy')):
                continue
            
            # Read bytes directly from zip
            with file.open('rb') as f_src:
                with open(os.path.join(dst_dir, f'{file.name[:-6]}{count_files}{file.suffix}'), 'wb') as f_dst:
                    shutil.copyfileobj(f_src, f_dst)

            count_files += 1


def copy_data_from_zip(
    zip_path: str,
    base_source_path: str,
    source_paths: list[str], 
    base_destination_path: str,
    destination_paths: list[str],
    num_img_dirs: int = 10,
    num_img_subdirs: int = 5,
) -> None:
    [
        os.makedirs(os.path.join(base_destination_path, dst_path), exist_ok=True)
        for dst_path in destination_paths
    ]

    global count_files

    with ZipFile(zip_path, 'r') as zip_ref:
        root = ZipPath(zip_ref)

        for src_rel, dst_rel in zip(source_paths, destination_paths):
            full_src_path = root / base_source_path / src_rel
            full_dst_path = os.path.join(base_destination_path, dst_rel)

            print(f"Copying from {full_src_path} to {full_dst_path}...")

            count_files = 0

            count_dirs = 0
            for folder in full_src_path.iterdir():
                if not folder.is_dir():
                    continue
                if count_dirs >= num_img_dirs:
                    break
                count_dirs += 1

                # Copy individual subdirectory images
                copy_files(folder, full_dst_path, num_img_subdirs)


if __name__ == "__main__":
    zip_path = './pupil_diameter/PupilDiameter.zip'
    base_source_path = "eyedentify/eyedentify"
    source_paths = [
        # 'left_eyes',
        # "right_eyes",
        'left_eyes_depth_maps',
        "right_eyes_depth_maps"
    ]
    base_destination_path = './pupil_diameter/PupilDiameter'
    destination_paths = source_paths  # same structure

    copy_data_from_zip(
        zip_path, 
        base_source_path,
        source_paths,
        base_destination_path,
        destination_paths,
        num_img_dirs=10,        # top-level dirs
        num_img_subdirs=5,      # subdirs in each dir
    )

