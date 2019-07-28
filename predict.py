# -*- coding: utf-8 -*-

"""
CPTN network predict process

run command: python predict.py [path] [mode] [running_mode]

args:
    path: image path (when mode='one')
    mode: one or random
    running_mode: cpu or gpu
"""

import copy
import math
import os
import random
import shutil
import sys

import cv2
import numpy as np
import pyximport
import torch

import handler
import lib.nms
import lib.utils
import net.network as Net

pyximport.install()

anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]

IMG_TEST_ROOT = "./mtwi_2018/image_test"
TEST_RESULT = './test_result'
THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
NEIGHBOURS_MIN_DIST = 50
MIN_ANCHOR_BATCH = 2
MODEL = ""


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_box(boxes, image_size):
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, image_size[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, image_size[0]-1)
    return boxes


def fit_y(cx, cy, x1, x2):
    """Use 1-d polynomial fitting to estimate y1, y2.

    args:
        cx: center x
        cy: center y
        x1: point 1
        x2: point 2
    """
    len(cx) != 0
    # if X only include one point, the function will get line y=Y[0]
    if np.sum(cx == cx[0]) == len(cx):
        return cy[0], cy[0]
    p = np.poly1d(np.polyfit(cx, cy, 1))
    return p(x1), p(x2)


def get_text_recs(text_proposals, image_size, scores=0):
    lines = np.zeros((len(text_proposals), 8), np.float32)
    for index, tp_indices in enumerate(text_proposals):
        line_boxes = np.array(tp_indices)
        cx = (line_boxes[:, 0] + line_boxes[:, 2]) / 2   # center x
        cy = (line_boxes[:, 1] + line_boxes[:, 3]) / 2   # center y
        z1 = np.polyfit(cx, cy, 1)    # 1-d polynomial fitting (y=kx+b)
        x0 = np.min(line_boxes[:, 0])  # min x
        x1 = np.max(line_boxes[:, 2])  # max x
        offset = (line_boxes[0, 2] - line_boxes[0, 0]) * 0.5
        # left-top, right-top
        lt_y, rt_y = fit_y(line_boxes[:, 0], line_boxes[:, 1], x0 + offset, x1 - offset)
        # left-base, right-base
        lb_y, rb_y = fit_y(line_boxes[:, 0], line_boxes[:, 3], x0 + offset, x1 - offset)
        lines[index, 0] = x0
        lines[index, 1] = min(lt_y, rt_y)   # min y on the top
        lines[index, 2] = x1
        lines[index, 3] = max(lb_y, rb_y)   # max y on the base
        lines[index, 4] = scores
        lines[index, 5] = z1[0]             # k, b (z1: y=kx+b)
        lines[index, 6] = z1[1]
        height = np.mean((line_boxes[:, 3] - line_boxes[:, 1]))
        lines[index, 7] = height + 2.5

    text_recs = np.zeros((len(lines), 9), np.float32)
    index = 0
    for line in lines:
        b1 = line[6] - line[7] / 2
        b2 = line[6] + line[7] / 2
        x1 = line[0]
        y1 = line[5] * line[0] + b1 # left-top
        x2 = line[2]
        y2 = line[5] * line[2] + b1 # right-top
        x3 = line[0]
        y3 = line[5] * line[0] + b2 # left-base
        x4 = line[2]
        y4 = line[5] * line[2] + b2 # right-base
        dx = x2 - x1
        dy = y2 - y1
        width = np.sqrt(dx * dx + dy * dy)  # text line width
        height = y3 - y1                    # text line height
        temp = height * dy / width
        x = np.fabs(temp * dx / width)
        y = np.fabs(temp * dy / width)
        if line[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        # in clock-wise order
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x4
        text_recs[index, 5] = y4
        text_recs[index, 6] = x3
        text_recs[index, 7] = y3
        text_recs[index, 8] = line[4]
        index = index + 1

    text_recs = clip_box(text_recs, image_size)

    return text_recs


def meet_v_iou(y1, y2, h1, h2):
    def overlaps_v(y1, y2, h1, h2):
        return max(0, y2-y1+1)/min(h1, h2)

    def size_similarity(h1, h2):
        return min(h1, h2)/max(h1, h2)

    return overlaps_v(y1, y2, h1, h2) >= 0.6 and size_similarity(h1, h2) >= 0.6


def gen_test_pair(image_dir, test_num=10):
    image_list = os.listdir(image_dir)
    if test_num > 0:
        random_list = random.sample(image_list, test_num)
    else:
        random_list = image_list
    test_pair = []
    for i in random_list:
        name, _ = os.path.splitext(i)
        image_path = os.path.join(image_dir, i)
        test_pair.append(image_path)
    return test_pair


def get_anchor_h(anchor, v):
    vc = v[int(anchor[7]), 0, int(anchor[5]), int(anchor[6])]
    vh = v[int(anchor[7]), 1, int(anchor[5]), int(anchor[6])]
    cya = anchor[5] * 16 + 7.5
    ha = anchor_height[int(anchor[7])]
    cy = vc * ha + cya
    h = math.pow(10, vh) * ha
    return h


def succession(v, anchors=[]):
    texts = []
    for i, anchor in enumerate(anchors):
        neighbours = []
        neighbours.append(i)
        center_x1 = (anchor[2] + anchor[0]) / 2
        h1 = get_anchor_h(anchor, v)
        # find i's neighbour
        for j in range(0, len(anchors)):
            if j == i:
                continue
            center_x2 = (anchors[j][2] + anchors[j][0]) / 2
            h2 = get_anchor_h(anchors[j], v)
            # less than 50 pixel between each anchor
            if abs(center_x1 - center_x2) < NEIGHBOURS_MIN_DIST and \
                    meet_v_iou(max(anchor[1], anchors[j][1]), min(anchor[3], anchors[j][3]), h1, h2):
                neighbours.append(j)
        if len(neighbours) != 0:
            texts.append(neighbours)

    need_merge = True
    while need_merge:
        need_merge = False
        # combine again
        for i, line in enumerate(texts):
            if len(line) == 0:
                continue
            for index in line:
                for j in range(i+1, len(texts)):
                    if index in texts[j]:
                        texts[i] += texts[j]
                        texts[i] = list(set(texts[i]))
                        texts[j] = []
                        need_merge = True

    result = []
    for text in texts:
        if len(text) < MIN_ANCHOR_BATCH:
            continue
        local = []
        for j in text:
            local.append(anchors[j])
        result.append(local)
    return result



def predict_one(image_path, net):
    """Predict only one image.
    """
    img = cv2.imread(image_path)
    img = handler.scale_image_only(img)
    image = copy.deepcopy(img)
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, :, :, :]
    image = torch.Tensor(image)
    y, score, side = net(image, val=True)
    result = []
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            for k in range(score.shape[2]):
                if score[i, j, k, 1] > THRESHOLD:
                    result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))

    for_nms = []
    for box in result:
        pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
        for_nms.append([pt[0], pt[1], pt[2], pt[3], box[3], box[0], box[1], box[2]])
    for_nms = np.array(for_nms, dtype=np.float32)
    nms_result = lib.nms.cpu_nms(for_nms, NMS_THRESHOLD)

    out_nms = []
    for i in nms_result:
        out_nms.append(for_nms[i, 0:8])

    connect = succession(y, out_nms)
    texts = get_text_recs(connect, img.shape)

    for box in texts:
        box = np.array(box)
        print(box)
        lib.utils.draw_ploy_4pt(img, box[0:8], thickness=2)

    _, basename = os.path.split(image_path)
    cv2.imwrite(TEST_RESULT + '/infer_' + basename, img)

    for i in nms_result:
        vc = y[int(for_nms[i, 7]), 0, int(for_nms[i, 5]), int(for_nms[i, 6])]
        vh = y[int(for_nms[i, 7]), 1, int(for_nms[i, 5]), int(for_nms[i, 6])]
        cya = for_nms[i, 5] * 16 + 7.5
        ha = anchor_height[int(for_nms[i, 7])]
        cy = vc * ha + cya
        h = math.pow(10, vh) * ha
        lib.utils.draw_box_2pt(img, for_nms[i, 0:4])
    _, basename = os.path.split(image_path)
    cv2.imwrite(TEST_RESULT + '/infer_anchor_' + basename, img)


def random_test(net):
    """Test images in the folder randomly.
    """
    test_pair = gen_test_pair(IMG_TEST_ROOT, 0)
    #print(test_pair)
    if os.path.exists(TEST_RESULT):
        shutil.rmtree(TEST_RESULT)

    os.mkdir(TEST_RESULT)

    for t in test_pair:
        im = cv2.imread(t)
        im = handler.scale_image_only(im)
        image = copy.deepcopy(im)
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis, :, :, :]
        image = torch.Tensor(image)
        v, score, side = net(image, val=True)
        result = []
        for i in range(score.shape[0]):
            for j in range(score.shape[1]):
                for k in range(score.shape[2]):
                    if score[i, j, k, 1] > THRESHOLD:
                        result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))

        for_nms = []
        for box in result:
            pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
            for_nms.append([pt[0], pt[1], pt[2], pt[3], box[3], box[0], box[1], box[2]])
        for_nms = np.array(for_nms, dtype=np.float32)
        nms_result = lib.nms.cpu_nms(for_nms, NMS_THRESHOLD)

        out_nms = []
        for i in nms_result:
            out_nms.append(for_nms[i, 0:8])

        connect = succession(v, out_nms)
        texts = get_text_recs(connect, im.shape)

        for i in nms_result:
            vc = v[int(for_nms[i, 7]), 0, int(for_nms[i, 5]), int(for_nms[i, 6])]
            vh = v[int(for_nms[i, 7]), 1, int(for_nms[i, 5]), int(for_nms[i, 6])]
            cya = for_nms[i, 5] * 16 + 7.5
            ha = anchor_height[int(for_nms[i, 7])]
            cy = vc * ha + cya
            h = math.pow(10, vh) * ha
            lib.utils.draw_box_2pt(im, for_nms[i, 0:4])
            #im = other.draw_frame(im, int(for_nms[i, 6]), cy, h)

        for box in texts:
            box = np.array(box)
            print(box)
            lib.utils.draw_ploy_4pt(im, box[0:8], thickness=2)
        cv2.imwrite(os.path.join(TEST_RESULT, os.path.basename(t)), im)
        

if __name__ == '__main__':
    if not os.path.exists(TEST_RESULT):
        os.mkdir(TEST_RESULT)
    running_mode = sys.argv[2]  # CPU or GPU
    print("Mode: %s" % running_mode)
    net = Net.CTPN()
    if running_mode == 'cpu':
        net.load_state_dict(torch.load(MODEL, map_location=running_mode))
    else:
        net.load_state_dict(torch.load(MODEL))
    #print(net)
    net.eval()
    if sys.argv[1] == 'random':
        random_test(net)
    else:
        path = sys.argv[1]
        predict_one(path, net)
