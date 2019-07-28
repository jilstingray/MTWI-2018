# -*- coding: utf-8 -*-

"""
CPTN training process.
"""

import configparser
import copy
import datetime
import logging
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

import evaluate
import handler
import lib.utils
import lib.utils
import net.loss as Loss
import net.network as Net

DRAW_PREFIX = './anchor_draw'
DATASET = './mtwi_2018'
MODEL_SAVE_PATH = './model'


def loop_file(path):
    files = []
    l = os.listdir(path)
    for f in l:
        files.append(os.path.join(path, f))
    return files


def get_train_val():
    train_img_list = []
    test_img_list = []
    train_txt_list = []
    test_txt_list = []
    train_img_path = os.path.join(DATASET, 'image_train')
    test_img_path = os.path.join(DATASET, 'image_test')
    train_txt_path = os.path.join(DATASET, 'txt_train')
    test_txt_path = os.path.join(DATASET, 'txt_test')
    train_img = loop_file(train_img_path)
    train_txt = loop_file(train_txt_path)
    test_img = loop_file(test_img_path)
    test_txt = loop_file(test_txt_path)
    train_img_list += train_img
    test_img_list += test_img
    train_txt_list += train_txt
    test_txt_list += test_txt
    return train_img_list, train_txt_list, test_img_list, test_txt_list 


def plot_loss(train_loss_list=[], test_loss_list=[]):
    """Draw loss function curve.

    Args:
        train_loss_list
        test_loss_list
    """
    x1 = range(0, len(train_loss_list))
    x2 = range(0, len(test_loss_list))
    y1 = train_loss_list
    y2 = test_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('train loss vs. iterators')
    plt.ylabel('train loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('test loss vs. iterators')
    plt.ylabel('test loss')
    plt.savefig("test_train_loss.jpg")


if __name__ == '__main__':
    # log
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    log_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    log_handler = logging.FileHandler(os.path.join(log_dir, log_name), 'w')
    log_format = logging.Formatter('%(asctime)s: %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)
    
    # config
    cf = configparser.ConfigParser()
    cf.read('./config')
    gpu_id = cf.get('global', 'gpu_id')
    epoch = cf.getint('global', 'epoch')
    val_batch_size = cf.getint('global', 'val_batch')
    logger.info('Total epoch: {0}'.format(epoch))
    using_cuda = cf.getboolean('global', 'using_cuda')
    disp_name = cf.getboolean('global', 'display_file_name')
    disp_iter = cf.getint('global', 'display_iter')
    val_iter = cf.getint('global', 'val_iter')
    save_iter = cf.getint('global', 'save_iter')
    lr_front = cf.getfloat('parameter', 'lr_front')
    lr_behind = cf.getfloat('parameter', 'lr_behind')
    change_epoch = cf.getint('parameter', 'change_epoch') - 1
    logger.info('Learning rate: {0}, {1}, change epoch: {2}'.format(lr_front, lr_behind, change_epoch + 1))
    print('GPU ID (available if use cuda): {0}'.format(gpu_id))
    print('Training epoches: {0}'.format(epoch))
    print('Use CUDA: {0}'.format(using_cuda))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    no_grad = ['cnn.VGG_16.convolution1_1.weight', 'cnn.VGG_16.convolution1_1.bias',
                'cnn.VGG_16.convolution1_2.weight', 'cnn.VGG_16.convolution1_2.bias']

    # create directory of saved models
    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    # load CTPN PyTorch network
    net = Net.CTPN()
    for name, value in net.named_parameters():
        if name in no_grad:
            value.requires_grad = False
        else:
            value.requires_grad = True
    #for name, value in net.named_parameters():
    #   print('name: {0}, grad: {1}'.format(name, value.requires_grad))
    net.load_state_dict(torch.load('./model/vgg16.model'))
    #net.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    lib.utils.init_weight(net)
    if using_cuda:
        net.cuda()
    net.train()

    # print network construct
    #print(net)

    # training loss
    get_loss = Loss.CTPN_Loss(using_cuda=using_cuda)
    train_img_list, train_txt_list, val_img_list, val_txt_list = get_train_val()
    total_iter = len(train_img_list)
    print("number of training images: %s" % len(train_img_list))
    print("number of validation images: %s" % len(val_img_list))
    train_loss_list = []
    test_loss_list = []
    for i in range(epoch):
        if i >= change_epoch:
            lr = lr_behind
        else:
            lr = lr_front
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        #optimizer = optim.Adam(net.parameters(), lr=lr)
        iterator = 1
        total_loss = 0
        total_cls_loss = 0
        total_y_reg_loss = 0
        total_s_reg_loss = 0
        start_time = time.time()
        random.shuffle(train_img_list)
        #print(random_im_list)
        for j in train_img_list:
            root, file_name = os.path.split(i)
            root, _ = os.path.split(root)
            name, _ = os.path.splitext(file_name)
            txt_name = name + '.txt'
            txt_path = os.path.join(root, 'txt_train', txt_name)
            if not os.path.exists(txt_path):
                print('txt file of image {0} not exists.'.format(j))
                continue
            txt = handler.read_txt_file(txt_path)
            #print("processing image %s" % os.path.join(image_root1, img))
            image = cv2.imread(j)
            if image is None:
                iterator += 1
                continue
            image, txt = handler.scale_image(image, txt)
            tensor_image = image[np.newaxis, :, :, :]
            tensor_image = tensor_image.transpose((0, 3, 1, 2))
            if using_cuda:
                tensor_image = torch.FloatTensor(tensor_image).cuda()
            else:
                tensor_image = torch.FloatTensor(tensor_image)
            pred_y, score, side = net(tensor_image)
            del tensor_image
            # transform bbox txt to anchor txt for training
            positive = []
            negative = []
            y_reg = []
            side_reg = []
            visual_image = copy.deepcopy(image)
            try:
                # loop all bbox in one image
                for k in txt:
                    # generate anchors from a bbox
                    txt_anchor, visual_image = \
                        lib.generate_anchor.generate_anchor(image, k, draw_image_box=visual_image)
                    positive1, negative1, y_reg1, side_reg1 = lib.utils.tag_anchor(txt_anchor, score, k)
                    positive += positive1
                    negative += negative1
                    y_reg += y_reg1
                    side_reg += side_reg1
            except:
                print("Warning: image %s raise error!" % j)
                batch_size += 1
                continue
            if len(y_reg) == 0 or len(positive) == 0 or len(side_reg) == 0:
                iterator += 1
                continue
            cv2.imwrite(os.path.join(DRAW_PREFIX, file_name), visual_image)
            optimizer.zero_grad()
            loss, cls_loss, y_reg_loss, s_reg_loss = get_loss(score, pred_y, side, positive, negative, y_reg, side_reg)
            loss.backward()
            optimizer.step()
            iterator += 1
            # save gpu memory by transferring loss to float
            total_loss += float(loss)
            total_cls_loss += float(cls_loss)
            total_y_reg_loss += float(y_reg_loss)
            total_s_reg_loss += float(s_reg_loss)

            if iterator % disp_iter == 0:
                end_time = time.time()
                total_time = end_time - start_time
                print('Epoch: {2}/{3}, Iteration: {0}/{1}, loss: {4}, \
                    cls_loss: {5}, v_reg_loss: {6}, o_reg_loss: {7}, {8}' \
                        .format(iterator, total_iter, i, epoch, total_loss / disp_iter,
                        total_cls_loss / disp_iter, total_y_reg_loss / disp_iter, total_s_reg_loss / disp_iter, j))
                
                # log
                logger.info('Epoch: {2}/{3}, Iteration: {0}/{1}'.format(iterator, total_iter, i, epoch))
                logger.info('loss: {0}'.format(total_loss / disp_iter))
                logger.info('classification loss: {0}'.format(total_cls_loss / disp_iter))
                logger.info('vertical regression loss: {0}'.format(total_y_reg_loss / disp_iter))
                logger.info('side-refinement regression loss: {0}'.format(total_s_reg_loss / disp_iter))
                train_loss_list.append(total_loss)
                total_loss = 0
                total_cls_loss = 0
                total_y_reg_loss = 0
                total_s_reg_loss = 0
                start_time = time.time()

            # validation during the training process
            if iterator % val_iter == 0:
                net.eval()
                logger.info(
                    'Start evaluate at {0} epoch {1} iteration.'.format(i, iterator))
                val_loss = evaluate.evalue(net, get_loss, val_batch_size, using_cuda, logger, val_img_list)
                logger.info('End evaluate.')
                net.train()
                start_time = time.time()
                test_loss_list.append(val_loss)

            # save model
            if iterator % save_iter == 0:
                print('Model saved at ./model/ctpn-{0}-{1}.model'.format(i, iterator))
                torch.save(net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'ctpn-{0}-{1}.model'.format(i, iterator)))
        print('Model saved at ./model/ctpn-{0}-end.model'.format(i))
        torch.save(net.state_dict(), os.path.join(
            MODEL_SAVE_PATH, 'ctpn-{0}-end.model'.format(i)))
        # draw loss function curve
        plot_loss(train_loss_list, test_loss_list)
