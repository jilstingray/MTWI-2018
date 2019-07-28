# -*- coding: utf-8 -*-

"""
Dataset handler: reorganize training dataset, create dataset using LMDB (optional) 
    and some necessary data processing functions.
If you've got the original MTWI_2018 dataset from Aliyun, try to use reorganize_dataset() to reorganize it.
The LMDB dataset generation procedure is optional.
Some functions (scaling, etc.) in this file are necessary for training
"""

import codecs
import os
import sys

import cv2
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import lib.utils as utils
import net.network as Net


def prehandler(src_path, out_path):
    txt_list = os.listdir(src_path)
    for f in txt_list:
        src = open(os.path.join(src_path, f), encoding='utf-8')
        out = open(os.path.join(out_path, f), 'w+')
        while 1:
            line = src.readline()
            if not line:
                break
            else:
                temp = line.split(',')
                temp[2], temp[3], temp[6], temp[7] = temp[6], temp[7], temp[2], temp[3]
                for i in range(7):
                    print(format(temp[i] + ","), end="", file=out)
                print(format(temp[7]), file=out)
        src.close()
        out.close()
    

def read_txt(path):
    """Read txt files.

    Each element of txt is an 8-dimensional vector, which stores the vertex coords of the text box.
    """
    result = []
    fp = open(path, 'r', encoding='utf-8')
    for line in fp.readlines():
        pt = line.split(',')
        box = [int(round(float(pt[i]))) for i in range(8)]
        result.append(box)
    fp.close()
    return result


def create_dataset(image_dir, txt_dir, out_dir):
    """Create MTWI-2018 dataset.

    args: 
        image_dir
        txt_dir
        out_dir
    """
    image_list = os.listdir(image_dir)
    image_path_list = []
    txt_path_list = []
    for i in image_list:
        name, _ = os.path.splitext(i)
        txt_name = name + '.txt'
        txt_dir = os.path.join(txt_dir, txt_name) 
        if not os.path.exists(txt_dir):
            print('Labels of image {0} not found.'.format(i))
        image_path_list.append(os.path.join(image_dir, i))
        txt_path_list.append(txt_dir)
    assert len(image_path_list) == len(txt_path_list)
    create_dataset(out_dir, image_path_list, txt_path_list)


def scale_image(image, txt, shortest_side=600):
    """Scale the image and coords of the boxes.

    return:
        image
        scale_txt
    """
    height = image.shape[0]
    width = image.shape[1]
    # get the zoom ratio of the image and scale the original image
    scale = float(shortest_side)/float(min(height, width))
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    
    # scale the image size to 600*600
    if image.shape[0] < image.shape[1] and image.shape[0] != 600:
        image = cv2.resize(image, (600, image.shape[1]))
    elif image.shape[0] > image.shape[1] and image.shape[1] != 600:
        image = cv2.resize(image, (image.shape[0], 600))
    elif image.shape[0] != 600:
        image = cv2.resize(image, (600, 600))
    # recalculate the zoom ratio of the width and height
    h_scale = float(image.shape[0])/float(height)
    w_scale = float(image.shape[1])/float(width)
    scale_txt = []
    # scale coords of the box
    for box in txt:
        scale_box = []
        for i in range(len(box)):
            if i % 2 == 0:
                scale_box.append(int(int(box[i]) * w_scale))
            else:
                scale_box.append(int(int(box[i]) * h_scale))
        scale_txt.append(scale_box)
    return image, scale_txt


def scale_image_only(image, shortest_side=600):
    height = image.shape[0]
    width = image.shape[1]
    scale = float(shortest_side)/float(min(height, width))
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    if image.shape[0] < image.shape[1] and image.shape[0] != 600:
        image = cv2.resize(image, (600, image.shape[1]))
    elif image.shape[0] > image.shape[1] and image.shape[1] != 600:
        image = cv2.resize(image, (image.shape[0], 600))
    elif image.shape[0] != 600:
        image = cv2.resize(image, (600, 600))
    return image


# 
def check_image(image):
    """Check image integrity.
    """
    if image is None:
        return False
    height, width = image.shape[0], image.shape[1]
    if height * width == 0:
        return False
    return True


def write_data(env, data):
    with env.begin(write=True) as e:
        for name, image in data.items():
            #print(type(name))  # <class 'str'>
            #print(type(image)) # <class 'bytes'>
            e.put(name.encode(), str(image).encode())


def list2str(input_list):
    """List (coords) to string.
    """
    result = []
    for box in input_list:
        if not len(box) % 8 == 0:
            return '', False
        result.append(','.join('%s' %i for i in box))
    return '|'.join(result), True


def create_dataset(output_dir, image_list, txt_list):
    """Create LMDB dataset (optional).
    """
    assert len(image_list) == len(txt_list)
    network = Net.VGG_16()
    num = len(image_list)
    # create directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # restrict size of dataset file (1.5GB limit)
    env = lmdb.Environment(output_dir, map_size=1610612736)
    cache = {}
    counter = 1
    for i in range(num):
        image_path = image_list[i]
        txt = read_txt(txt_list[i])
        # check existance of image path
        if not os.path.exists(image_path):
            print("{0} not found.".format(image_path))
            continue
        # check existance of txt file
        if len(txt) == 0:
            print("labels of {0} not found.".format(image_path))
            continue
        # read image
        image = cv2.imread(image_path)
        # check integrity of image
        if not check_image(image):
            print('image {0} is not valid.'.format(image_path))
            continue
        
        # scale the image and coords
        image, txt = scale_image(image, txt)
        # list to string
        txt_str = list2str(txt)
        if not txt_str[1]:
            print("labels of {0} are not valid.".format(image_path))
            continue
        
        # save the image and the txt into 'env'
        image_key = 'image-%09d' % counter
        txt_key = 'txt-%09d' % counter
        cache[image_key] = utils.np_img2base64(image, image_path)
        cache[txt_key] = txt_str[0]
        counter += 1
        if counter % 100 == 0:
            write_data(env, cache)
            cache.clear()
            print('written {0}/{1}'.format(counter, num))

    cache['num'] = str(counter - 1)

    write_data(env, cache)
    print('create dataset with {0} image.'.format(counter - 1))


class LMDB_dataset(Dataset):
    def __init__(self, root, transformer=None):
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print("cannot create lmdb from root {0}.".format(root))
        with self.env.begin(write=False) as e:
            self.data_num = int(e.get('num'))
        self.transformer = transformer

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        assert index <= len(self), 'Index out of range.'
        index += 1
        with self.env.begin(write=False) as e:
            image_key = 'image-%09d' % index
            image_base64 = e.get(image_key)
            image = utils.base642img(image_base64)
            txt_key = 'txt-%09d' % index
            txt = str(e.get(txt_key))
        return image, txt


def reorganize_dataset(image_dir, txt_dir):
    all_image = os.listdir(image_dir)
    all_label = os.listdir(txt_dir)
    all_image.sort()
    all_label.sort()
    count = 0
    for image_name, label_name in zip(all_image, all_label):
        image = Image.open(image_dir + '/' + image_name)
        image = np.array(image)

        # if image doesn't have readable RGB channels, abandon
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


if __name__ == '__main__':
    image_dir = './mtwi_2018/image_train'
    txt_dir = './mtwi_2018/txt_train'
    output_dir = 'data'
    reorganize_dataset(image_dir, txt_dir)
    # OPTIONAL: create LMDB dataset
    #create_dataset_mtwi(image_dir, txt_dir, output_dir)
