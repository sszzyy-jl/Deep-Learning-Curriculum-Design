# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import os
import os.path as osp

import imgviz  # 图片可视化
import numpy as np

import labelme


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default="D:/Dataset/road_dataset/annotations", help="input annotated directory")
    parser.add_argument("--output_dir", default="D:/Dataset/road_dataset", help="output dataset directory")
    parser.add_argument("--labels", default="D:/Dataset/road_dataset/label.txt", help="labels file")
    args = parser.parse_args()
    args.noviz = False

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClassPNG"))
    # 创建目录
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationClassVisualization")
        )
    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    # 解析标签文件，为每个类别分配一个唯一的类别ID，并将类别名称与ID映射关系存储在字典中
    for i, line in enumerate(open(args.labels).readlines()):  # 逐行阅读文件
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:  # 第一行一般要忽略
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:  # 第二行是背景
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    # 输出文件的路径，包存了类别名称

    # 将类别名称写入指定的文件中，并将文件路径输出到控制台上
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)
    # 使用 glob.glob 函数获取指定文件夹下所有以 .json 结尾的文件路径
    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)


        label_file = labelme.LabelFile(filename=filename)

        # 使用 osp.basename(filename) 获取文件名，然后通过 osp.splitext 函数获取基本文件名。
        base = osp.splitext(osp.basename(filename))[0]
        # 这部分代码用于构建三种不同格式的输出文件路径，分别是图像文件、分割标签文件（npy 格式）、分割标签文件（PNG 格式）
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_lbl_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".npy"
        )
        out_png_file = osp.join(
            args.output_dir, "SegmentationClassPNG", base + ".png"
        )
        #  构建了可视化分割标签的输出文件路径。它将可视化文件保存在 args.output_dir 目录下的 "SegmentationClassVisualization" 子目录中
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )

        with open(out_img_file, "wb") as f:  # wb为二进制文件
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        # 将标签文件中的图像数据转换为图像数组（通常是 NumPy 数组）

        # 函数labelme.utils.shapes_to_label将根据提供的参数根据标签文件中的形状信息创建一个标签图像lbl

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,  # 它告诉函数要创建一个与原始图像相同大小的标签图像。
            shapes=label_file.shapes,  # 这是标签文件中包含的形状信息。通常，这些形状代表了图像中不同类别的区域。
            label_name_to_value=class_name_to_id,  # 它将类别名称映射到类别值。这是为了将不同的类别标签与数字值相关联，以便在标签图像中表示它们。
        )
        labelme.utils.lblsave(out_png_file, lbl)  # 这行代码使用 labelme.utils.lblsave 函数将生成的标签图像保存为一个 PNG 文件

        np.save(out_lbl_file, lbl) # 这行代码使用 np.save 函数将标签图像 lbl 保存为一个 NumPy 数组文件

        # 用于可视化标签图像并保存可视化结果
        if not args.noviz:
            viz = imgviz.label2rgb(  # viz 是可视化结果的图像数据,使用 imgviz.label2rgb 函数生成，将标签图像 lbl 转换成彩色图像
                lbl,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",  # img 是原始图像，font_size 设置了标签名称的字体大小，label_names 包含类别的名称，loc 设置了标签位置（"rb" 表示右下角）。
            )
            imgviz.io.imsave(out_viz_file, viz)


if __name__ == "__main__":
    main()