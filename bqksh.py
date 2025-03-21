import json
import cv2
import numpy as np
from tqdm import tqdm
import os

fill_color = (255, 255, 255)
root_dir = 'D:/Dataset/road_dataset'


def visualize_one(label_name):
    with open(root_dir + '/annotations' + '/' + label_name + '.json', 'r') as obj:
        dict = json.load(obj)
    img = cv2.imread(root_dir + '/images' + '/' + label_name + '.jpg')
    for label in dict['shapes']:
        points = np.array(label['points'], dtype=np.int32)
        black_img = np.zeros(img.shape)
        cv2.polylines(black_img, [points], isClosed=True, color=fill_color, thickness=1)
        cv2.fillPoly(black_img, [points], color=fill_color)

    cv2.imwrite(root_dir + '/labels' + '/' + label_name + '.jpg', black_img)


if __name__ == '__main__':
    os.mkdir(root_dir + '/labels')
    for i in tqdm(os.listdir(os.path.join(root_dir, "annotations"))):
        label_name = i[:-5]
        visualize_one(label_name)