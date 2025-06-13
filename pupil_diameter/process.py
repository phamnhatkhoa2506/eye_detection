import numpy as np

if __name__ == "__main__":
    data = np.load('./pupil_diameter/PupilDiameter/left_eyes_depth_maps/frame_0.npy')

    print(data.shape)