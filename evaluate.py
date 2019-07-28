# -*- coding: utf-8 -*-

"""
Model evaluation 
"""

import copy
import os
import random
import time

import cv2
import numpy as np
import torch

import handler
import lib.utils
import lib.utils


def evalue(net, criterion, batch_size, using_cuda, logger, image_list):
    # get image list randomly
    random_list = random.sample(image_list, batch_size)

    total_loss = 0
    total_cls_loss = 0      # classification loss
    total_v_reg_loss = 0    # vertical parameters' regression loss (height, y-axis coords)
    total_o_reg_loss = 0    # side-refinement loss of bbox

    # start evaluation
    start_time = time.time()
    for i in random_list:
        root, file_name = os.path.split(i)
        root, _ = os.path.split(root)
        name, _ = os.path.splitext(file_name)
        # get image's corresponding txt file
        txt_name = name + '.txt'
        txt_path = os.path.join(root, "txt_test", txt_name)
        if not os.path.exists(txt_path):
            print('txt file of image {0} not exists.'.format(txt_path))
            continue
        # read txt labels
        txt = handler.read_txt(txt_path)     
        # read image
        image = cv2.imread(i)
        if image is None:
            batch_size -= 1
            continue

        image, txt = handler.scale_image(image, txt)
        tensor_image = image[np.newaxis, :, :, :]
        tensor_image = tensor_image.transpose((0, 3, 1, 2))
        # using CUDA
        if using_cuda:
            tensor_image = torch.FloatTensor(tensor_image).cuda()
        # using CPU
        else:
            tensor_image = torch.FloatTensor(tensor_image)

        # network forwarding
        vertical_pred, score, side_refinement = net(tensor_image)
        del tensor_image
        positive = []
        negative = []
        vertical_reg = []
        side_refinement_reg = []
        visual_image = copy.deepcopy(image)

        try:
            for box in txt:
                # generate anchor
                txt_anchor, visual_image = lib.utils.gen_anchor(image, box, draw_image_box=visual_image)
                # tag anchor
                positive1, negative1, vertical_reg1, side_refinement_reg1 = lib.utils.tag_anchor(txt_anchor, score, box)
                positive += positive1
                negative += negative1
                vertical_reg += vertical_reg1
                side_refinement_reg += side_refinement_reg1
        except:
            print("warning: image %s raise error!" % i)
            batch_size -= 1
            continue

        if len(vertical_reg) == 0 or len(positive) == 0 or len(side_refinement_reg) == 0:
            batch_size -= 1
            continue

        # calculating losses
        loss, txts_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive,
                                                           negative, vertical_reg, side_refinement_reg)
        total_loss += float(loss)
        total_cls_loss += float(total_cls_loss)
        total_v_reg_loss += float(v_reg_loss)
        total_o_reg_loss += float(o_reg_loss)

    # end evaluation
    end_time = time.time()
    total_time = end_time - start_time

    if batch_size == 0:
        print('evaluation failed.')
        return 1
    
    print('\n--- start evaluation ---')
    print('loss: {0}'.format(total_loss / float(batch_size)))
    logger.info('evaluate loss: {0}'.format(total_loss / float(batch_size)))

    print('classification loss: {0}'.format(total_cls_loss / float(batch_size)))
    logger.info('evaluate vertical regression loss: {0}'.format(total_v_reg_loss / float(batch_size)))

    print('vertical regression loss: {0}'.format(total_v_reg_loss / float(batch_size)))
    logger.info('evaluate side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_size)))

    print('side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_size)))
    logger.info('evaluate side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_size)))

    print('{1} iterations for {0} seconds.'.format(total_time, batch_size))
    print('--- end evaluation ---\n')
    return total_loss
