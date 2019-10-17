# -*- coding: utf-8 -*-

"""
This file contains some utilites.
"""

import base64
import copy
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image


def draw_box_4pt(image, pts, color=(0, 255, 0), thickness=2):
    """Draw a text box (4 points).
    """
    if not isinstance(pts[0], int):
        pts = [int(pts[i]) for i in range(8)]
    image = cv2.line(image, (pts[0], pts[1]), (pts[2], pts[3]), 255, thickness)
    image = cv2.line(image, (pts[0], pts[1]), (pts[6], pts[7]), 255, thickness)
    image = cv2.line(image, (pts[4], pts[5]), (pts[6], pts[7]), 255, thickness)
    image = cv2.line(image, (pts[4], pts[5]), (pts[2], pts[3]), 255, thickness)
    return image


def draw_poly_4pt(image, pts, color=(0, 255, 255), thickness=1):
    """Draw polygonal curves (4 points).
    """
    points = np.array([[pts[0], pts[1]], [pts[2], pts[3]], [pts[4], pts[5]], [pts[6], pts[7]]], np.int32)
    #print(points)
    points = points.reshape((-1, 1, 2))
    return cv2.polylines(image, [points], True, color, thickness)


def draw_box_2pt(image, pts, color=(0, 255, 0), thickness=1):
    """Draw a rectangle through the diagonal (2 points).
    """
    if not isinstance(pts[0], int):
        pts = [int(pts[i]) for i in range(4)]
    image = cv2.rectangle(image, (pts[0], pts[1]), (pts[2], pts[3]), color, thickness=thickness)
    return image


def draw_slice(image, x, cy, h, anchor_width=16, color=(0, 255, 0), thickness=1):
    """Draw slices.
    """
    x_left = x * anchor_width
    x_right = (x + 1) * anchor_width - 1
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_base = int(cy + (float(h) - 1) / 2.0)
    pt = [x_left, y_top, x_right, y_base]
    return draw_box_2pt(image, pt, color=color, thickness=thickness)


def np2base64(np_image, path):
    """Numpy to base64
    """
    image = cv2.imencode(os.path.splitext(path)[1], np_image)[1]
    image = np.squeeze(image, 1)
    b64code = base64.b64encode(image)
    return b64code


def base642img(b64code):
    """base64 to image
    """
    missing_padding = 4 - len(b64code) % 4
    if missing_padding:
        b64code += b'=' * missing_padding
    np_image = np.fromstring(base64.b64decode(b64code), dtype=np.uint8)
    image = cv2.imdecode(np_image, cv2.COLOR_RGB2BGR)
    return image


def get_y(pos1, pos2, x, form):
    """Calculate the y corresponding to x on the line connected by point 1 & 2

    args:
        pos1: point 1 position
        pos2: point 2 position
        x
        form
    """
    if not isinstance(pos1[0], float) or not isinstance(pos2[0], float):
        pos1 = [float(pos1[i]) for i in range(len(pos1))]
        pos2 = [float(pos2[i]) for i in range(len(pos2))]
    if not isinstance(x, float):
        x = float(x)
    if (pos1[0] - pos2[0]) == 0:
        return -1
    return form(((pos1[1] - pos2[1])/(pos1[0] - pos2[0])) * (x - pos1[0]) + pos1[1])


def get_range(start, end):
    start = int(start)
    end = int(end)
    if start > end:
        return range(end, start)
    else:
        return range(start, end)


def trans_to_2pt(x, cy, h, anchor_width=16):
    x_left = x * anchor_width
    x_right = (x + 1) * anchor_width - 1
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_base = int(cy + (float(h) - 1) / 2.0)
    return [x_left, y_top, x_right, y_base]


def init_weight(net):
    for i in range(len(net.rnn.blstm.lstm.all_weights)):
        for j in range(len(net.rnn.blstm.lstm.all_weights[0])):
            torch.nn.init.normal_(net.rnn.blstm.lstm.all_weights[i][j], std=0.01)

    torch.nn.init.normal_(net.FC.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.FC.bias, val=0)

    torch.nn.init.normal_(net.vertical_coordinate.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.vertical_coordinate.bias, val=0)

    torch.nn.init.normal_(net.score.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.score.bias, val=0)

    torch.nn.init.normal_(net.side_refinement.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.side_refinement.bias, val=0)


def gen_anchor(image, txt, anchor_width=16, draw_image_box=None):
    """Generate the anchor.

    args:
        image
        txt
        anchor_width
    
    return:
        list of tuple(position, center_y, height)
    """
    result = []
    box = [float(txt[i]) for i in range(len(txt))]  # text box
    left_anchor_id = int(math.floor(max(min(box[0], box[2]), 0) / anchor_width))
    right_anchor_id = int(math.ceil(min(max(box[6], box[4]), image.shape[1]) / anchor_width))
    if right_anchor_id * 16 + 15 > image.shape[1]:
        right_anchor_id -= 1

    # combine the left and right x-axis coords of each anchor into a pair
    pairs = [(i * anchor_width, (i + 1) * anchor_width - 1) for i in range(left_anchor_id, right_anchor_id)]

    # calculate the y-axis coords of the top and base bounds of anchors in the text box
    y_tops, y_bases = anchor_y(image, pairs, box)

    # return list of tuple(position, center_y, height)
    for i in range(len(pairs)):
        if pairs != [] and i < len(y_tops):
            position = int(pairs[i][0] / anchor_width)
            cy = (float(y_bases[i]) + float(y_tops[i])) / 2.0
            h = y_bases[i] - y_tops[i] + 1
            result.append((position, cy, h))
            # 绘制ground truth anchor
            draw_image_box = draw_slice(draw_image_box, position, cy, h)
        draw_image_box = draw_box_4pt(draw_image_box, box, color=(0, 0, 255), thickness=1)
    return result, draw_image_box


def anchor_y(raw_image, pairs, pt):
    """Calculate the y-axis coords of the top and base bounds of anchors in the text box.

    args:
        raw_image: input image
        pairs: for example: [(0, 15), (16, 31), ...]
        pt: coords of the vertexs of anchors
    
    return: 
        top and base y-axis coords of anchors
    """
    image = copy.deepcopy(raw_image)
    y_tops = []
    y_bases = []
    height = image.shape[0]

    # set channel 0 as mask
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j, 0] = 0

    # draw the text box
    image = draw_box_4pt(image, pt, color=(255, 0, 0))

    is_top = False
    is_base = False
    for i in range(len(pairs)):
        # top
        for y in range(0, height - 1):
            for x in range(pairs[i][0], pairs[i][1] + 1):
                if image[y, x, 0] == 255:
                    y_tops.append(y)
                    is_top = True
                    break
            if is_top is True:
                break
        # base
        for y in range(height - 1, -1, -1):
            for x in range(pairs[i][0], pairs[i][1] + 1):
                if image[y, x, 0] == 255:
                    y_bases.append(y)
                    is_base = True
                    break
            if is_base is True:
                break
        is_top = False
        is_base = False

    return y_tops, y_bases


def get_anchor_in_image(image_path, txt_path):
    """Find all anchors in the image.

    args:
        image_path: *.jpg 
        label_path: *.txt
    
    return:
        list of anchors
    """
    image = Image.open(image_path)
    image = np.array(image)
    txt = open(txt_path, "r", encoding='utf-8')
    result = []

    for line in txt.readlines():
        line = line.split(',')
        # 去掉text标签
        label = [float(line[i]) for i in range(8)]
        result.append(gen_anchor(image, label))
    
    txt.close()
    return result


def get_iou(cy1, h1, cy2, h2):
    """Calculate intersection over union (IoU).

    args:
        cy1: center y
        h1: height
        cy2: center y
        h2: height
    """
    y_top1, y_base1 = y_range(cy1, h1)
    y_top2, y_base2 = y_range(cy2, h2)
    offset = min(y_top1, y_top2)
    y_top1 = y_top1 - offset
    y_top2 = y_top2 - offset
    y_base1 = y_base1 - offset
    y_base2 = y_base2 - offset
    line = np.zeros(max(y_base1, y_base2) + 1)
    for i in range(y_top1, y_base1 + 1):
        line[i] += 1
    for j in range(y_top2, y_base2 + 1):
        line[j] += 1
    union = np.count_nonzero(line, 0)
    intersec = line[line == 2].size
    return float(intersec)/float(union)


def y_range(cy, h):
    """Get the y-axiz coords of the top and base bound of a rectangle.

    args:
        cy: center y
        h: height
    """
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_base = int(cy + (float(h) - 1) / 2.0)
    return y_top, y_base


def valid_anchor(cy, h, height):
    """Validate the anchor.

    args:
        cy: center y of the anchor
        h: height of the anchor
        height: height of the slices
    """
    top, base = y_range(cy, h)
    if top < 0:
        return False
    if base > (height * 16 - 1):
        return False
    return True


def tag_anchor(anchor, cnn_output, box):
    """Tag the anchor.

    Calculate vertical regression and side-refinement loss.
    """
    anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]  # from 11 to 273, divide 0.7 each time
    height = cnn_output.shape[2]
    width = cnn_output.shape[3]
    positive = []
    negative = []
    y_reg = []
    side_reg = []
    left_x = min(box[0], box[6])
    right_x = max(box[2], box[4])
    left_side = False
    right_side = False

    for a in anchor:
        if a[0] >= int(width - 1):
            continue

        if left_x in range(a[0] * 16, (a[0] + 1) * 16):
            left_side = True
        else:
            left_side = False

        if right_x in range(a[0] * 16, (a[0] + 1) * 16):
            right_side = True
        else:
            right_side = False

        iou = np.zeros((height, len(anchor_height)))
        temp_positive = []
        for i in range(iou.shape[0]):
            for j in range(iou.shape[1]):
                if not valid_anchor((float(i) * 16.0 + 7.5), anchor_height[j], height):
                    continue
                iou[i][j] = get_iou((float(i) * 16.0 + 7.5), anchor_height[j], a[1], a[2])

                if iou[i][j] > 0.7:
                    temp_positive.append((a[0], i, j, iou[i][j]))
                    if left_side:
                        o = (float(left_x) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_reg.append((a[0], i, j, o))
                    if right_side:
                        o = (float(right_x) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_reg.append((a[0], i, j, o))

                if iou[i][j] < 0.5:
                    negative.append((a[0], i, j, iou[i][j]))

                if iou[i][j] > 0.5:
                    vc = (a[1] - (float(i) * 16.0 + 7.5)) / float(anchor_height[j])
                    vh = math.log10(float(a[2]) / float(anchor_height[j]))
                    y_reg.append((a[0], i, j, vc, vh, iou[i][j]))

        if len(temp_positive) == 0:
            max_position = np.where(iou == np.max(iou))
            temp_positive.append((a[0], max_position[0][0], max_position[1][0], np.max(iou)))

            if left_side:
                o = (float(left_x) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_reg.append((a[0], max_position[0][0], max_position[1][0], o))
            if right_side:
                o = (float(right_x) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_reg.append((a[0], max_position[0][0], max_position[1][0], o))

            if np.max(iou) <= 0.5:
                vc = (a[1] - (float(max_position[0][0]) * 16.0 + 7.5)) / float(anchor_height[max_position[1][0]])
                vh = math.log10(float(a[2]) / float(anchor_height[max_position[1][0]]))
                y_reg.append((a[0], max_position[0][0], max_position[1][0], vc, vh, np.max(iou)))

        positive += temp_positive
        
    return positive, negative, y_reg, side_reg
