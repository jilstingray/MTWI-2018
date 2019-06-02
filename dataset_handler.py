# -*- coding: utf-8 -*-

"""
Dataset handler: create dataset using LMDB
"""

import codecs
import os
import sys
import cv2
import lmdb
from torch.utils.data import Dataset
import lib.utils as utils
import net.network as Net

# 数据集预处理
def dataset_prehandler(src_path, out_path):
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
    

# 读入txt标签文件
def read_txt_file(path):
    """
    读入给定路径下的txt, 一个txt对应一张图
    txt的每个元素为一个8维向量, 储存box（文字框）的顶点坐标
    """
    result = []
    fp = open(path, 'r', encoding='utf-8')
    # 读入标签
    for line in fp.readlines():
        pt = line.split(',')
        box = [int(round(float(pt[i]))) for i in range(8)]
        result.append(box)
    fp.close()
    return result


# 建立MTWI-2018数据集
def create_dataset_mtwi(image_dir, txt_dir, output_dir):
    """
    args: 
        image_dir: 图片文件夹路径
        txt_dir: 文件夹路径
        output_dir: 输出路径
    """
    image_list = os.listdir(image_dir)
    image_path_list = []
    txt_path_list = []
    for i in image_list:
        name, _ = os.path.splitext(i)
        txt_name = name + '.txt'
        txt_path = os.path.join(txt_dir, txt_name) 
        if not os.path.exists(txt_path):
            print('Labels of image {0} not found.'.format(i))
        image_path_list.append(os.path.join(image_dir, i))
        txt_path_list.append(txt_path)
    assert len(image_path_list) == len(txt_path_list)
    create_dataset(output_dir, image_path_list, txt_path_list)


# 缩放图像
def scale_image(image, txt, shortest_side=300):
    """
    此函数的作用是缩放图像, 同时相应的边框的坐标也要缩放
    return: image, txt scale
    """
    height = image.shape[0]
    width = image.shape[1]
    # 求得图像的缩放系数, 对原始图像进行缩放
    scale = float(shortest_side)/float(min(height, width))
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    
    # 将图片尺寸缩放为300*300
    if image.shape[0] < image.shape[1] and image.shape[0] != 300:
        image = cv2.resize(image, (300, image.shape[1]))
    elif image.shape[0] > image.shape[1] and image.shape[1] != 300:
        image = cv2.resize(image, (image.shape[0], 300))
    elif image.shape[0] != 300:
        image = cv2.resize(image, (300, 300))
    # 重新计算宽高各自的缩放比, 应该很接近
    h_scale = float(image.shape[0])/float(height)
    w_scale = float(image.shape[1])/float(width)
    scale_txt = []
    # 边框坐标同时缩放
    for box in txt:
        scale_box = []
        for i in range(len(box)):
            if i % 2 == 0:
                scale_box.append(int(int(box[i]) * w_scale))
            else:
                scale_box.append(int(int(box[i]) * h_scale))
        scale_txt.append(scale_box)
    return image, scale_txt


def scale_image_only(image, shortest_side=300):
    height = image.shape[0]
    width = image.shape[1]
    scale = float(shortest_side)/float(min(height, width))
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    if image.shape[0] < image.shape[1] and image.shape[0] != 300:
        image = cv2.resize(image, (300, image.shape[1]))
    elif image.shape[0] > image.shape[1] and image.shape[1] != 300:
        image = cv2.resize(image, (image.shape[0], 300))
    elif image.shape[0] != 300:
        image = cv2.resize(image, (300, 300))
    return image


def check_image(image):
    if image is None:
        return False
    height, width = image.shape[0], image.shape[1]
    if height * width == 0:
        return False
    return True


def write_cache(env, data):
    with env.begin(write=True) as e:
        for name, image in data.items():
            #print(type(name))  # <class 'str'>
            #print(type(image)) # <class 'bytes'>
            e.put(name.encode(), str(image).encode())


# 边框坐标列表转化为字符输出
def list2str(input_list):
    result = []
    for box in input_list:
        if not len(box) % 8 == 0:
            return '', False
        result.append(','.join('%s' %i for i in box))
    return '|'.join(result), True


# 建立数据集
def create_dataset(output_dir, image_list, txt_list):
    assert len(image_list) == len(txt_list)
    network = Net.VGG_16()
    num = len(image_list)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    env = lmdb.Environment(output_dir, map_size=524288000)  # 500MB
    cache = {}
    counter = 1
    for i in range(num):
        image_path = image_list[i]
        txt = read_txt_file(txt_list[i]) 
        if not os.path.exists(image_path):
            print("{0} not found.".format(image_path))
            continue

        if len(txt) == 0:
            print("Labels of {0} not found.".format(image_path))
            continue

        image = cv2.imread(image_path)
        if not check_image(image):
            print('Image {0} is not valid.'.format(image_path))
            continue
        
        # 缩放图片和坐标
        image, txt = scale_image(image, txt)
        # 坐标转化为字符
        txt_str = list2str(txt)
        if not txt_str[1]:
            print("Labels of {0} are not valid.".format(image_path))
            continue
        
        # 把图片和坐标存入env
        image_key = 'image-%09d' % counter
        txt_key = 'txt-%09d' % counter
        cache[image_key] = utils.np_img2base64(image, image_path)
        cache[txt_key] = txt_str[0]
        counter += 1
        if counter % 100 == 0:
            write_cache(env, cache)
            cache.clear()
            print('Written {0}/{1}'.format(counter, num))

    cache['num'] = str(counter - 1)

    write_cache(env, cache)
    print('Create dataset with {0} image.'.format(counter - 1))


class LMDB_dataset(Dataset):
    def __init__(self, root, transformer=None):
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print("Cannot create lmdb from root {0}.".format(root))
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
            image = utils.base642np_image(image_base64)
            txt_key = 'txt-%09d' % index
            txt = str(e.get(txt_key))
        return image, txt


if __name__ == '__main__':
    image_dir = './mtwi_2018/image_train'
    txt_dir = './mtwi_2018/txt_train'
    output_dir = 'data'
    create_dataset_mtwi(image_dir, txt_dir, output_dir)