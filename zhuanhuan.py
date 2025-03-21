# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

input_folder = 'input_folder'  # 输入文件夹路径
output_folder = 'output_folder'  # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for file_name in os.listdir(input_folder):
    if file_name.startswith('CE3_BMYK_PCAML-C-'):
        file_path = os.path.join(input_folder, file_name)

        # 打开文件并读取数据
        with open(file_path, 'rb') as f:
            data = f.read()

        # 提取图像数据部分并转换为numpy数组
        byte_array = bytearray(data[len(data) - 2352 * 1728 * 3:])
        numpy_array = np.frombuffer(byte_array, dtype=np.uint8).reshape((1728, 2352, 3))

        # 转换颜色空间和调整大小
        numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        numpy_array = cv2.resize(numpy_array, (2352 // 2, 1728 // 2))

        # 生成输出文件路径
        output_file_path = os.path.join(output_folder, file_name.replace('.2C', '.jpg'))

        # 保存图像
        cv2.imwrite(output_file_path, numpy_array)

print("转换完成！")

