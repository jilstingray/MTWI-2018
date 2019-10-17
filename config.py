# -*- coding: utf-8 -*-

"""
Create config file.

Check your GPU first and modify the content below.
"""

import configparser

if __name__ == '__main__':
    cp = configparser.ConfigParser()
    cp.add_section('global')
    # GPU settings
    cp.set('global', 'using_cuda', 'True')
    cp.set('global', 'gpu_id', '0')
    # epoch settings
    cp.set('global', 'epoch', '10')
    # iterator settings
    cp.set('global', 'display_file_name', 'False')
    cp.set('global', 'display_iter', '10')
    cp.set('global', 'val_batch', '50')
    cp.set('global', 'val_iter', '50')
    cp.set('global', 'save_iter', '100')
    # parameter settings
    cp.add_section('parameter')
    cp.set('parameter', 'lr_front', '0.001')
    cp.set('parameter', 'lr_behind', '0.0001')
    cp.set('parameter', 'change_epoch', '9')
    with open('./config', 'w+') as fp:
        cp.write(fp)
