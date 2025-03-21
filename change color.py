import cv2
import numpy as np
# 读取图片
image = cv2.imread('D:/Dataset/road_dataset/SegmentationClassPNG/0011.png')
# 定义黑色的范围
lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([30, 30, 30], dtype=np.uint8)
# 创建掩码，将黑色部分标记为白色
black_mask = cv2.inRange(image, lower_black, upper_black)
# 将黑色部分替换为白色
image[black_mask != 0] = [255, 255, 255]
# 保存处理后的图片
cv2.imwrite('output_image.jpg', image)