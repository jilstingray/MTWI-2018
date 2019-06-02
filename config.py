# -*- coding: utf-8 -*-

"""
Create config file.
"""

import configparser  # for Python 3.x

if __name__ == '__main__':
    cp = configparser.ConfigParser()
    cp.add_section('global')
    # using CUDA/CPU in training process
    cp.set('global', 'using_cuda', 'True')
    # epoch (1个epoch表示过了1遍训练集中的所有样本)
    cp.set('global', 'epoch', '10')
    # GPU ID (MUST check your computer's settings first)
    cp.set('global', 'gpu_id', '0')
    cp.set('global', 'display_file_name', 'False')
    # display iterator
    cp.set('global', 'display_iter', '10')
    # validation batch size
    cp.set('global', 'val_batch', '50')
    # validation iterator
    cp.set('global', 'val_iter', '50')
    # save iterator
    cp.set('global', 'save_iter', '100')
    # parameter settings
    cp.add_section('parameter')
    cp.set('parameter', 'lr_front', '0.001')
    cp.set('parameter', 'lr_behind', '0.0001')
    cp.set('parameter', 'change_epoch', '9')
    with open('./config', 'w+') as fp:
        cp.write(fp)
