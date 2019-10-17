# -*- coding: utf-8 -*-

"""CTPN implementation based on PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_16(nn.Module):
    """VGG16 without fully connected (FC) layers.

    Low-level feature extraction.
    """
    def __init__(self):
        super(VGG_16, self).__init__()
        # conv1
        self.convolution1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.convolution1_2 = nn.Conv2d(64, 64, 3, padding=1)
        # max pooling
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        # conv2
        self.convolution2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.convolution2_2 = nn.Conv2d(128, 128, 3, padding=1)
        # max pooling
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        # conv3
        self.convolution3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.convolution3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convolution3_3 = nn.Conv2d(256, 256, 3, padding=1)
        # max pooling
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        # conv4
        self.convolution4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.convolution4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # max pooling
        self.pooling4 = nn.MaxPool2d(2, stride=2)
        # conv5
        self.convolution5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_3 = nn.Conv2d(512, 512, 3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.convolution1_1(x), inplace=True)
        x = F.relu(self.convolution1_2(x), inplace=True)
        x = self.pooling1(x)
        x = F.relu(self.convolution2_1(x), inplace=True)
        x = F.relu(self.convolution2_2(x), inplace=True)
        x = self.pooling2(x)
        x = F.relu(self.convolution3_1(x), inplace=True)
        x = F.relu(self.convolution3_2(x), inplace=True)
        x = F.relu(self.convolution3_3(x), inplace=True)
        x = self.pooling3(x)
        x = F.relu(self.convolution4_1(x), inplace=True)
        x = F.relu(self.convolution4_2(x), inplace=True)
        x = F.relu(self.convolution4_3(x), inplace=True)
        x = self.pooling4(x)
        x = F.relu(self.convolution5_1(x), inplace=True)
        x = F.relu(self.convolution5_2(x), inplace=True)
        x = F.relu(self.convolution5_3(x), inplace=True)
        return x


class Im2col(nn.Module):
    """Convert the feature map of the VGG16 output to a vector.

    Convert convolution operation to matrix multiplication.
    """
    def __init__(self, kernel_size, stride, padding):
        super(Im2col, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        height = x.shape[2]
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x = x.reshape((x.shape[0], x.shape[1], height, -1))
        return x


class BLSTM(nn.Module):
    """Bi-directional Long Short-Term Memory Network
    """
    def __init__(self, channel, hidden_unit, bidirectional=True):
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(channel, hidden_unit, bidirectional=bidirectional)

    def forward(self, x):
        x = x.transpose(1, 3)   # the batch size of x must be 1
        recurrent, _ = self.lstm(x[0])
        recurrent = recurrent[np.newaxis, :, :, :]
        recurrent = recurrent.transpose(1, 3)
        return recurrent


class CTPN(nn.Module):
    """Connectionist Text Proposal Network
    
    Connect VGG16 with BLSTM.
    
    Returns:
        pred_y
        score
        side_refinement
    """
    def __init__(self):
        super(CTPN, self).__init__()
        # VGG16 CNN
        self.cnn = nn.Sequential()
        self.cnn.add_module('VGG_16', VGG_16())
        # BLSTM RNN
        self.rnn = nn.Sequential()
        self.rnn.add_module('Im2col', Im2col((3, 3), (1, 1), (1, 1)))
        self.rnn.add_module('blstm', BLSTM(3 * 3 * 512, 128))
        # 全连接层
        self.FC = nn.Conv2d(256, 512, 1)
        # 输出2*10个参数, 2个参数分别表示anchor的height和center_y, 10表示anchor的尺寸个数
        self.vertical_coordinate = nn.Conv2d(512, 2 * 10, 1)
        # 输出2*10个分数, 2表示有无字符, 10表示anchor的尺寸个数
        self.score = nn.Conv2d(512, 2 * 10, 1)
        # 输出10个参数, 表示anchor的水平偏移, 用于调整文本框水平边缘精度, 10表示anchor的尺寸个数
        self.side_refinement = nn.Conv2d(512, 10, 1)

    def forward(self, x, val=False):
        x = self.cnn(x)
        x = self.rnn(x)
        x = self.FC(x)
        x = F.relu(x, inplace=True)
        pred_y = self.vertical_coordinate(x)
        score = self.score(x)
        if val:
            score = score.reshape((score.shape[0], 10, 2, score.shape[2], score.shape[3]))
            score = score.squeeze(0)
            score = score.transpose(1, 2)
            score = score.transpose(2, 3)
            score = score.reshape((-1, 2))
            #score = F.softmax(score, dim=1)
            score = score.reshape((10, pred_y.shape[2], -1, 2))
            pred_y = pred_y.reshape((pred_y.shape[0], 10, 2, pred_y.shape[2],
                                                   pred_y.shape[3]))
            pred_y = pred_y.squeeze(0)
        side_refinement = self.side_refinement(x)
        return pred_y, score, side_refinement
