import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.file_handler import img_iter


# Load the images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
path = r'C:\Users\omri_\OneDrive\Documents\neurogrametry_data\dogarden'
images = img_iter(path)

# Define the known intrinsic matrix and distortion coefficients
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# dist_coeffs = np.array([k1, k2, p1, p2, k3])

orb = cv2.ORB_create()
fundamental_matrices = []
is_first = True
image_list = []
points = []
descriptions = []
position = []
#initialization of 3d plot matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

for image in images:
    # Convert the image to grayscale
    image_list.append(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the feature points in the image
    # kp = orb.detect(gray, None)
    # points.append(np.array([p.pt for p in kp], dtype=np.float32).reshape((-1, 1, 2)))
    # kp, des = orb.compute(gray, kp)
    # descriptions.append(des)
    points.append(cv2.goodFeaturesToTrack(gray, 100, 0.01, 10))
    if is_first:
        is_first = False
        continue
    image_list = image_list[-2:]
    points = points[-2:]
    pts1 = points[0]
    pts2 = points[1]
    img1 = image_list[0].copy()
    img2 = image_list[1].copy()
    # show corresponding points on both images
    for i in range(len(pts1)):
        random_color = np.random.randint(0, 255, (3,)).tolist()
        cv2.circle(img1, (int(pts1[i][0][0]), int(pts1[i][0][1])), 5, random_color, -1)
        cv2.circle(img2, (int(pts2[i][0][0]), int(pts2[i][0][1])), 5, random_color, -1)
    # resize images to fit on screen
    img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
    # create a side-by-side image
    img3 = np.hstack((img1, img2))
    cv2.imshow('image3', img3)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


    # Check if the points were found
    if points is not None:
        # Estimate the essential matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2,cv2.FM_8POINT)
        # decompose F to R, t
        S = cv2.decomposeEssentialMat(F)
        position.append(S[2])
        # Recover the rotation and translation from the essential matrix
        # _, R, t, mask = cv2.recoverPose(E, points, camera_matrix)
        # Append the extrinsic matrix to the list
        # extrinsic_matrices.append(np.hstack((R, t)))
        #plot from position:
        ax.scatter(position[-1][0], position[-1][1], position[-1][2], c='r', marker='o')
        plt.pause(0.05)
    else:
        print("No feature points found in image:", image)

plt.show()