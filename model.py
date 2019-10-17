# -*- coding: utf-8 -*-

"""Generate revised VGG16 model.
"""

import sys

import torch
import torchvision.models as models

import net.network as Net


def gen_VGG16(net, output_path):
    """Generate VGG16 model.
    """

    vgg_16 = models.vgg16(pretrained=False)
    #print(vgg_16)

    # load VGG16 pretrained model
    vgg_16.load_state_dict(torch.load('./model/vgg16-397923af.pth'))

    # generate
    pretrained_dict = vgg_16.state_dict()
    model_dict = net.state_dict()
    check_list = [[], []]
    for i in range((2 + 2 + 3 + 3 + 3) * 2):
        check_list[0].append(list(pretrained_dict.keys())[i])
        check_list[1].append(list(model_dict.keys())[i])
    backbone_dict = {}
    for i in range((2 + 2 + 3 + 3 + 3) * 2):
        backbone_dict[check_list[1][i]] = pretrained_dict[check_list[0][i]]
    model_dict.update(backbone_dict)
    #for i in range((2 + 2 + 3 + 3 + 3) * 2):
    #   print((model_dict[model_dict.keys()[i]] == pretrained_dict[pretrained_dict.keys()[i]]).all())
    
    # save model
    torch.save(model_dict, output_path)


if __name__ == '__main__':
    gen_VGG16(Net.CTPN(), './model/vgg16.model')
