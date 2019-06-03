# -*- coding: utf-8 -*-

"""
This file is to re-organize the MTWI_2018 dataset labels and generate CTPN anchors.
If you've got the original MTWI_2018 dataset from Aliyun, try to set reorganize_dataset() as main() to re-organize it.
	(X1, Y1): left_top; (X2, Y2): left_bottom
	(X3, Y3): right_bottom; (X4, Y4): right_top
anchor format: list of tuple(position, center_y, height)
Only coordinates are needed in this task.
"""

import copy
import math
import os
import cv2
import numpy as np
from PIL import Image
import lib.utils as utils

# 生成anchor
def generate_anchor(image, txt, anchor_width=16, draw_image_box=None):
    """
    args:
        image: input image
        txt: input MTWI_2018 labimage(without text)
        anchor_width: the width of the anchors
    return:
        list of tuple(position, center_y, height)
    """
    result = []
    box = [float(txt[i]) for i in range(len(txt))]  # text box

    # 得到文本框最左侧/最右侧的anchor的id
    left_anchor_id = int(math.floor(max(min(box[0], box[2]), 0) / anchor_width))
    right_anchor_id = int(math.ceil(min(max(box[6], box[4]), image.shape[1]) / anchor_width))

    # 极端情况: 最右侧anchor可能超出图像边界
    if right_anchor_id * 16 + 15 > image.shape[1]:
        right_anchor_id -= 1

    # 将每个anchor的左右x轴坐标组合成一对pair
    pairs = [(i * anchor_width, (i + 1) * anchor_width - 1) for i in range(left_anchor_id, right_anchor_id)]

    # 计算文本框中所有anchor的上下边界的y轴坐标
    y_top, y_bottom = cal_anchor_bound(image, pairs, box)

    # return list of tuple(position, center_y, height)
    for i in range(len(pairs)):
        if pairs != [] and i < len(y_top):
            position = int(pairs[i][0] / anchor_width)
            cy = (float(y_bottom[i]) + float(y_top[i])) / 2.0
            h = y_bottom[i] - y_top[i] + 1
            result.append((position, cy, h))
            # 绘制ground truth anchor
            draw_image_box = utils.draw_box_h_and_c(draw_image_box, position, cy, h)
        draw_image_box = utils.draw_box_4pt(draw_image_box, box, color=(0, 0, 255), thickness=1)
    return result, draw_image_box


# 计算文本框中所有anchor的上下边界的y轴坐标
def cal_anchor_bound(raw_image, pairs, pt):
    """
    args:
        raw_image: input image
        pairs: for example: [(0, 15), (16, 31), ...]
        pt: anchor's 4 corners' coordinates
    return: 
        anchor's top & bottom y-axis coordinates
    """
    image = copy.deepcopy(raw_image)
    y_top = []
    y_bottom = []
    height = image.shape[0]
    # set channel 0 as mask
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j, 0] = 0

    # 绘制文本框
    image = utils.draw_box_4pt(image, pt, color=(255, 0, 0))
    """
    pt = [int(i) for i in coord]
    image = cv2.line(image, (pt[0], pt[1]), (pt[2], pt[3]), 255, thickness=1)
    image = cv2.line(image, (pt[0], pt[1]), (pt[6], pt[7]), 255, thickness=1)
    image = cv2.line(image, (pt[4], pt[5]), (pt[6], pt[7]), 255, thickness=1)
    image = cv2.line(image, (pt[4], pt[5]), (pt[2], pt[3]), 255, thickness=1)
    """

    is_top = False
    is_bottom = False
    
    for i in range(len(pairs)):
        # 处理上边界
        for y in range(0, height - 1):
            for x in range(pairs[i][0], pairs[i][1] + 1):
                if image[y, x, 0] == 255:
                    y_top.append(y)
                    is_top = True
                    break
            if is_top is True:
                break
        
        # 处理下边界
        for y in range(height - 1, -1, -1):
            for x in range(pairs[i][0], pairs[i][1] + 1):
                if image[y, x, 0] == 255:
                    y_bottom.append(y)
                    is_bottom = True
                    break
            if is_bottom is True:
                break
        
        is_top = False
        is_bottom = False

    return y_top, y_bottom


# 从图像中找出所有anchor
def get_anchors_from_image(image_path, label_path):
    """
    args:
        image_path: *.jpg 
        label_path: *.txt
    return:
        list of anchors
    """
    image = Image.open(image_path)
    image = np.array(image)
    txt = open(label_path, "r", encoding='utf-8')
    result = []

    for line in txt.readlines():
        line = line.split(',')
        # 去掉text标签
        label = [float(line[i]) for i in range(8)]
        result.append(generate_anchor(image, label))
    
    txt.close()
    return result


# 整理数据集
def reorganize_dataset():
    image_dir = "./image_train"
    txt_dir = "./txt_train"
    all_image = os.listdir(image_dir)
    all_label = os.listdir(txt_dir)
    all_image.sort()
    all_label.sort()
    count = 0
    for image_name, label_name in zip(all_image, all_label):
        image = Image.open(image_dir + '/' + image_name)
        image = np.array(image)

        # if image doesn't have RGB channels, abandon
        if len(image.shape) < 3:
            print("Bad image: " + image_name)
            os.remove(image_dir + '/' + image_name)
            os.remove(txt_dir + '/' + label_name)

        else:
            os.rename(image_dir + '/' + image_name, image_dir + '/' + str(count) + ".jpg")
            os.rename(txt_dir + '/' + label_name, txt_dir + '/' + str(count) + ".txt")
            image_name = str(count) + ".jpg"
            label_name = str(count) + ".txt"
            count += 1
