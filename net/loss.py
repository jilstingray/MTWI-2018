# -*- coding: utf-8 -*-

"""
CTPN loss calculation.
"""

import random
import torch
import torch.nn as nn

class CTPN_Loss(nn.Module):
    def __init__(self, using_cuda=False):
        super(CTPN_Loss, self).__init__()
        self.Ns = 128
        self.ratio = 0.5
        self.lambda1 = 1.0
        self.lambda2 = 1.0
        self.Ls_cls = nn.CrossEntropyLoss()  # classification loss 
        self.Lv_reg = nn.SmoothL1Loss()  # vertical coordinate regression loss
        self.Lo_reg = nn.SmoothL1Loss()  # side-refinement regression loss
        self.using_cuda = using_cuda

    def forward(self, score, pred_y, side, positive, negative, y_reg, side_reg):
        """Forward propagation.

        args:
            score: prediction score
            pred_y: vertical coordinate prediction
            side: side refinement prediction
            positive: ground truth positive fine-scale box
            negativev: ground truth negative fine-scale box
            y_reg: ground truth vertical regression
            side_reg: ground truth side-refinement regression
        
        return:
            loss: total loss
            cls_loss: cross entropy loss
            y_reg_loss: vertical regression loss
            s_reg_loss: side-refinement regression loss
        """
        ## classification loss (using CrossEntropyLoss)
        cls_loss = 0.0
        positive_num = min(int(self.Ns * self.ratio), len(positive))
        negative_num = self.Ns - positive_num
        positive_batch = random.sample(positive, positive_num)
        negative_batch = random.sample(negative, negative_num)
        # using CUDA
        if self.using_cuda:
            for p in positive_batch:
                cls_loss += self.Ls_cls(score[0, p[2] * 2: ((p[2] + 1) * 2), p[1], p[0]].unsqueeze(0), torch.LongTensor([1]).cuda())
            for n in negative_batch:
                cls_loss += self.Ls_cls(score[0, n[2] * 2: ((n[2] + 1) * 2), n[1], n[0]].unsqueeze(0), torch.LongTensor([0]).cuda())
        # using CPU
        else:
            for p in positive_batch:
                cls_loss += self.Ls_cls(score[0, p[2] * 2: ((p[2] + 1) * 2), p[1], p[0]].unsqueeze(0), torch.LongTensor([1]))
            for n in negative_batch:
                cls_loss += self.Ls_cls(score[0, n[2] * 2: ((n[2] + 1) * 2), n[1], n[0]].unsqueeze(0), torch.LongTensor([0]))
        cls_loss = cls_loss / self.Ns

        ## vertical coordinate regression loss (using SmoothL1Loss)
        y_reg_loss = 0.0
        Nv = len(y_reg)
        # using CUDA
        if self.using_cuda:
            for v in y_reg:
                y_reg_loss += self.Lv_reg(pred_y[0, v[2] * 2: ((v[2] + 1) * 2), v[1], v[0]].unsqueeze(0), torch.FloatTensor([v[3], v[4]]).unsqueeze(0).cuda())
        # using CPU
        else:
            for v in y_reg:
                y_reg_loss += self.Lv_reg(pred_y[0, v[2] * 2: ((v[2] + 1) * 2), v[1], v[0]].unsqueeze(0), torch.FloatTensor([v[3], v[4]]).unsqueeze(0))
        y_reg_loss = y_reg_loss / float(Nv)

        ## side-refinement regression loss (using SmoothL1Loss)
        s_reg_loss = 0.0
        No = len(side_reg)
        # using CUDA
        if self.using_cuda:
            for s in side_reg:
                s_reg_loss += self.Lo_reg(side[0, s[2]: s[2] + 1, s[1], s[0]].unsqueeze(0), torch.FloatTensor([s[3]]).unsqueeze(0).cuda())
        # using CPU
        else:
            for s in side_reg:
                s_reg_loss += self.Lo_reg(side[0, s[2]: s[2] + 1, s[1], s[0]].unsqueeze(0), torch.FloatTensor([s[3]]).unsqueeze(0))
        s_reg_loss = s_reg_loss / float(No)

        ## total loss
        loss = cls_loss + y_reg_loss * self.lambda1 + s_reg_loss * self.lambda2
        return loss, cls_loss, y_reg_loss, s_reg_loss
