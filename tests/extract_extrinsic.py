import cv2
import numpy as np

from utils.file_handler import img_iter


# Load the images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
path = r'C:\Users\omri_\OneDrive\Documents\neurogrametry_data\dogarden'
images = img_iter(path)

# Define the known intrinsic matrix and distortion coefficients
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# dist_coeffs = np.array([k1, k2, p1, p2, k3])

# Define the image size
image_size = (images[0].shape[1], images[0].shape[0])

# Extract the extrinsic matrix of each image
extrinsic_matrices = []
for image in images:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the feature points in the image
    points = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
    # Check if the points were found
    if points is not None:
        # Estimate the essential matrix
        E, mask = cv2.findEssentialMat(points, camera_matrix)
        # Recover the rotation and translation from the essential matrix
        _, R, t, mask = cv2.recoverPose(E, points, camera_matrix)
        # Append the extrinsic matrix to the list
        extrinsic_matrices.append(np.hstack((R, t)))
    else:
        print("No feature points found in image:", image_path)