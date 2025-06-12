import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread('./segmentation/Seg_Dataset_Eye/IRIS_PUPIL_EYE/train/segmentation/0213_2_1_2_53_006.png', 0)

    for idx in np.unique(img):
        print(len(np.where(img == idx)[0]))

    plt.imshow(img, cmap="gray")
    plt.show()
