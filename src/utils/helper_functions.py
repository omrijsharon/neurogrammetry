import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import json


CAM2WORLD = np.array([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])


def read_yaml_file(file_path):
    with open(file_path) as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
    return d


def read_json_file(file_path):
    with open(file_path) as file:
            d = json.load(file)
    return d


def get_video_details(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height


def find_calib_file(video_file, camera_name, calib_dir=None):
    if calib_dir is None:
        calib_dir = r'C:\Users\omri_\OneDrive\Documents\repos\gyroflow\resources\camera_presets'
    fps, width, height = get_video_details(video_file)
    fps_str = "{:.2f}fps".format(fps)
    resolution = "{}x{}".format(width, height)
    existing_cameras = [cam_name.lower() for cam_name in os.listdir(calib_dir)]
    camera_name = camera_name.split()
    name_idx = -1
    for i, cam_name in enumerate(camera_name):
        if cam_name.lower() in existing_cameras:
            cam_name = cam_name.lower()
            name_idx = i
            break
    assert name_idx != -1, 'Camera name not found in calibration directory'
    del camera_name[name_idx]
    existing_models = [model_name.lower() for model_name in os.listdir(os.path.join(calib_dir, cam_name))]
    filtered_models = []
    for i, cam_name in enumerate(camera_name):
        existing_models = [model.lower() for model in existing_models if cam_name.lower() in model]
    existing_models = [model.lower() for model in existing_models if fps_str.lower() in model]
    existing_models = [model.lower() for model in existing_models if resolution in model]
    assert len(existing_models) == 1, 'More than one calibration file found: {}'.format(existing_models)
    calib_file = os.path.join(calib_dir, camera_name, existing_models[0]+'.json')
    return calib_file

def load_calibration_file(file_path, camera_name, resolution, fps):
    data = read_json_file(file_path)
    intrinsic_matrix = np.eye(3)
    intrinsic_matrix[0, 0] = data['f_x']
    intrinsic_matrix[0, 0] = data['f_y']
    intrinsic_matrix[0, 2] = data['c_x']
    intrinsic_matrix[1, 2] = data['c_y']
    dist_coeffs = np.array(data['distortion_coefficients'])
    return intrinsic_matrix, dist_coeffs


def undistoret_image(image, camera_matrix, dist_coeffs):
    return cv2.undistort(image, camera_matrix, dist_coeffs)


# Extract corresponding points from the two frames
def extract_correspondences(prev_frame, frame):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    prev_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    next_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    return prev_pts, next_pts


# Extract the extrinsic matrix from the two frames
def extract_extrinsic_matrix(prev_pts, next_pts):
    # Compute the fundamental matrix
    F, mask = cv2.findFundamentalMat(prev_pts, next_pts)
    # Compute the essential matrix
    K = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    E = K.T @ F @ K
    # Compute the rotation and translation matrices
    ret, R, t, mask = cv2.recoverPose(E, prev_pts, next_pts, K)
    return R, t


# use 8 point algorithm to find fundamental matrix
def find_fundamental_matrix(prev_pts, next_pts):
    F, mask = cv2.findFundamentalMat(prev_pts, next_pts,cv2.FM_8POINT)
    return F


# decompose F to R, t
def decompose_fundamental_matrix(F):
    S = cv2.decomposeEssentialMat(F)
    return S[1], S[2]


# resize images by factor
def resize_image(image, factor):
    return cv2.resize(image, (0, 0), fx=factor, fy=factor)


class FrameIterator:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.success, self.image = self.video.read()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.success:
            self.video.release()
            raise StopIteration
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.success, self.image = self.video.read()
        return image


class FramePrevIterator:
    def __init__(self, video_path, calib_path, nfeatures, scale_factor):
        self.intrinsic_matrix, self.dist_coeffs = load_calibration_file(calib_path)
        self.focal_length = self.intrinsic_matrix[0, 0]
        self.scale_factor = scale_factor
        self.pp = (int(self.intrinsic_matrix[0, 2]), int(self.intrinsic_matrix[1, 2]))
        self.video = cv2.VideoCapture(video_path)
        self.success, self.image = self.video.read()
        self.image = resize_image(self.image, self.scale_factor)
        self.prev_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # self.prev_gray = undistoret_image(self.prev_gray, self.intrinsic_matrix, self.dist_coeffs)
        self.gray = self.prev_gray.copy()
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.prev_gray, None)
        self.t = np.zeros((3, 1))
        self.R = np.eye(3)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.success:
            self.video.release()
            raise StopIteration
        self.prev_gray = self.gray
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # self.gray = undistoret_image(self.gray, self.intrinsic_matrix, self.dist_coeffs)
        kp2, des2 = self.orb.detectAndCompute(self.gray, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        prev_pts = np.float32([self.kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        next_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        self.kp1, self.des1 = kp2, des2
        self.success, self.image = self.video.read()
        self.image = resize_image(self.image, self.scale_factor)
        E, _ = cv2.findEssentialMat(next_pts, prev_pts, self.focal_length, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E, prev_pts, next_pts)
        self.t = self.t + self.R.dot(t)
        self.R = R.dot(self.R)
        return self.prev_gray, self.gray, prev_pts, next_pts, self.R, self.t

    def prev_gray(self):
        return self.prev_gray



if __name__ == '__main__':
    video_path = r'C:\Users\omri_\OneDrive\Documents\fpv\gopro\Yom_kipur_bike\GH010001.mp4'
    calib_path = r'C:\Users\omri_\PycharmProjects\neurogrammetry\config\dji_airunit_calib.yaml'
    find_calib_file(video_path, camera_name="gopro black hero7")
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))
    # initialization of a 3d plot with matplotlib:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim3d(-1, 1)
    # ax.set_ylim3d(-1, 1)
    # ax.set_zlim3d(-1, 1)
    # ax.view_init(azim=0, elev=90)

    for prev_frame, frame, prev_pts, pts, R, t in FramePrevIterator(video_path, calib_path, 2000, scale_factor=0.5):
        t_total = np.matmul(CAM2WORLD, t).flatten()
        ax.scatter(t_total[0], t_total[1], t_total[2], c='r', marker='o')
        plt.pause(0.001)
        # vstack the two frames
        stacked_frame = np.vstack((prev_frame, frame))
        # Draw the matches
        for i, (prev_pt, pt) in enumerate(zip(prev_pts, pts)):
            pt = (pt[0][0], pt[0][1] + prev_frame.shape[0])
            cv2.line(stacked_frame, tuple(prev_pt[0].astype(int)), tuple(np.array(pt).astype(int)), (0, 255, 0), 1)
            cv2.circle(stacked_frame, tuple(prev_pt[0].astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(stacked_frame, tuple(np.array(pt).astype(int)), 3, (0, 0, 255), -1)
        cv2.imshow('frame', stacked_frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
