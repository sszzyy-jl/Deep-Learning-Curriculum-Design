import math
import cv2 as cv
import numpy as np

src = cv.imread('output_image.jpg')
# 得到原图像的长宽
srcWidth = src.shape[1]
srcHeight = src.shape[0]
dist_coefs = np.array([0, 0, 0, 0, 0], dtype = np.float64)
camera_matrix = np.array([[srcWidth * 0.5, 0, srcWidth / 2], [0, srcWidth * 0.5, srcHeight / 2], [0, 0, 1]],
                         dtype = np.float64)    # 相机内参矩阵


newWidth = 500 # 新图像宽
newHeight = 800 # 新图像高
# 新相机内参，自己设定
newCam = np.array([[newWidth * 0.15, 0, newWidth / 2], [0, newWidth * 0.15, newHeight / 2], [0, 0, 1]])
invNewCam = np.linalg.inv(newCam)  # 内参逆矩阵
map_x = np.zeros((newHeight, newWidth), dtype=np.float32)
map_y = np.zeros((newHeight, newWidth), dtype=np.float32) # 用于存储图像坐标的数组

for k in range(10, 90, 10):
    pitch = 50 * 3.14 / 180
    print("pitch = ", pitch)
    R = np.array([[1, 0, 0], [0, math.sin(pitch), math.cos(pitch)], [0, -math.cos(pitch), math.sin(pitch)]])
    for i in range(map_x.shape[0]):
        for j in range(map_x.shape[1]):
            ray = np.dot(invNewCam, np.array([j, i, 1]).T)  # 像素转换为入射光线
            rr = np.dot(R, ray)  # 乘以旋转矩阵
            # 光线投影到像素点
            point, _ = cv.projectPoints(rr, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), camera_matrix,
                                        dist_coefs)
            map_x[i, j] = point[0][0][0]
            map_y[i, j] = point[0][0][1]
    dst = cv.remap(src, map_x, map_y, cv.INTER_LINEAR)  # 它接受输入图像src以及之前计算的map_x和map_y，然后进行图像的重映射。
    cv.imwrite("img_bev.jpg", dst)
    dst = cv.resize(dst, (2352 // 2, 1728 // 2))
    cv.imshow('dst', dst)
    cv.imshow('src', src)
    cv.waitKey()