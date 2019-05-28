# -*- coding: utf-8 -*-

"""
本文件需要修改: 这份文件是针对不同的数据集进行处理的
"""

import codecs
import os
import sys

import cv2
import lmdb  # lmdb: Lightning Memory-Mapped Database Manager
from torch.utils.data import Dataset

sys.path.append("../..")
import lib.utils as utils
import net.network as net


# 读入给定路径下的ground-truth
def read_gt_file(path, have_BOM=False):
    """
    输入: 路径, 是否有BOM
    输出: ground truth
    此函数的作用是读入给定路径下的ground-truth, 一个gt对应一张图
    gt的每个元素为一个8维向量, 储存box（文字框）的顶点坐标
    """
    result = []
    # 判断是否有BOM, 以此决定打开文本的格式
    if have_BOM:
        fp = codecs.open(path, 'r', 'utf-8-sig')
    else:
        fp = open(path, 'r', 'utf-8')
    # 一行行地读入结果
    for line in fp.readlines():
        pt = line.split(',')
        if have_BOM:
            box = [int(round(float(pt[i]))) for i in range(8)]
        else:
            box = [int(round(float(pt[i]))) for i in range(8)]
        result.append(box)
    fp.close()
    return result


# 建立ICDAR2015数据集
def create_dataset_icdar2015(img_root, gt_root, output_path):
    """
    args: 
        img_root: 图片文件夹路径
        gt_root: 文件夹路径, 输出路径
    输出: 图片路径 + gt路径
    """
    # 得到图片名字列表
    im_list = os.listdir(img_root)
    im_path_list = []
    gt_list = []
    for im in im_list:
        name, _ = os.path.splitext(im)
        gt_name = 'gt_' + name + '.txt'
        
        # os.path.join 的作用是拼接出完整的路径: 文件夹/name.txt
        gt_path = os.path.join(gt_root, gt_name) 
        if not os.path.exists(gt_path):
            print('Ground truth file of image {0} not exists.'.format(im))
        im_path_list.append(os.path.join(img_root, im))
        gt_list.append(gt_path)
    assert len(im_path_list) == len(gt_list)
    # 在输出路径下创建包含了图片的路径和gt的路径的文件
    create_dataset(output_path, im_path_list, gt_list)


# 缩放图像
def scale_img(img, gt, shortest_side=600):
    """
    此函数的作用是缩放图像, 同时相应的ground truth, 即边框的坐标也要缩放
    输出: 图片, ground truth scale
    """
    height = img.shape[0]
    width = img.shape[1]
    # 求得图像的缩放系数, 对原始图像进行缩放
    scale = float(shortest_side)/float(min(height, width))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    
    # 强行校准某一条边为600
    if img.shape[0] < img.shape[1] and img.shape[0] != 600:
        img = cv2.resize(img, (600, img.shape[1]))
    elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
        img = cv2.resize(img, (img.shape[0], 600))
    elif img.shape[0] != 600:
        img = cv2.resize(img, (600, 600))
    # 重新计算宽高各自的缩放比, 应该很接近
    h_scale = float(img.shape[0])/float(height)
    w_scale = float(img.shape[1])/float(width)
    scale_gt = []
    # 边框坐标同时缩放
    for box in gt:
        scale_box = []
        for i in range(len(box)):
            if i % 2 == 0:
                scale_box.append(int(int(box[i]) * w_scale))
            else:
                scale_box.append(int(int(box[i]) * h_scale))
        scale_gt.append(scale_box)
    return img, scale_gt


# 只对图像做缩放
def scale_img_only(img, shortest_side=600):
    height = img.shape[0]
    width = img.shape[1]
    scale = float(shortest_side)/float(min(height, width))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if img.shape[0] < img.shape[1] and img.shape[0] != 600:
        img = cv2.resize(img, (600, img.shape[1]))
    elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
        img = cv2.resize(img, (img.shape[0], 600))
    elif img.shape[0] != 600:
        img = cv2.resize(img, (600, 600))
    return img


# 检查图片是否正常
def check_img(img):
    if img is None:
        return False
    height, width = img.shape[0], img.shape[1]
    if height * width == 0:
        return False
    return True


# 将data写入env
def write_cache(env, data):
    """
    输入: 环境地址env, 数据data
    """
    with env.begin(write=True) as e:
        for i, l in data.iteritems():
            e.put(i, l)


# 边框坐标列表转化为字符输出
def box_list2str(l):
    result = []
    for box in l:
        if not len(box) % 8 == 0:
            return '', False
        result.append(','.join(box))
    return '|'.join(result), True


# 存储cache到输出路径
def create_dataset(output_path, img_list, gt_list):
    """
    输入: 输出路径, 图片地址列表, gt列表
    存储cache到输出路径, cache是一个存放了所有经过缩放的图片、经过了缩放的gt、图片总数的字典
    """
    assert len(img_list) == len(gt_list)
    net = net.VGG_16()
    num = len(img_list)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 打开env
    env = lmdb.open(output_path, map_size=1099511627776)
    # 字典初始化, 用来存储图片和文字和图片数目
    cache = {}
    counter = 1
    for i in range(num):
        img_path = img_list[i]
        gt = gt_list[i]
        if not os.path.exists(img_path):
            print("{0} is not exist.".format(img_path))
            continue

        if len(gt) == 0:
            print("Ground truth of {0} is not exist.".format(img_path))
            continue

        img = cv2.imread(img_path)
        if not check_img(img):
            print('Image {0} is not valid.'.format(img_path))
            continue
        
        # 缩放图片和gt
        img, gt = scale_img(img, gt)
        # gt转化为字符
        gt_str = box_list2str(gt)
        if not gt_str[1]:
            print("Ground truth of {0} is not valid.".format(img_path))
            continue
        
        # 把图片和gt一并存入cache字典
        img_key = 'image-%09d' % counter
        gt_key = 'gt-%09d' % counter
        cache[img_key] = utils.np_img2base64(img, img_path)
        cache[gt_key] = gt_str[0]
        counter += 1
        if counter % 100 == 0:
            write_cache(env, cache)
            cache.clear()
            print('Written {0}/{1}'.format(counter, num))

    cache['num'] = str(counter - 1)
    # 把cache字典写入env
    write_cache(env, cache)
    print('Create dataset with {0} image.'.format(counter - 1))


# 把env中的cache的信息读取出来
class LmdbDataset(Dataset):
    def __init__(self, root, transformer=None):
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print("Cannot create lmdb from root {0}.".format(root))
        with self.env.begin(write=False) as e:
            self.data_num = int(e.get('num'))
        self.transformer = transformer

    def __len__(self):
        # 读取图片数目
        return self.data_num

    def __getitem__(self, index):
        # 读取图片和gt
        assert index <= len(self), 'Index out of range.'
        index += 1
        with self.env.begin(write=False) as e:
            img_key = 'image-%09d' % index
            img_base64 = e.get(img_key)
            img = utils.base642np_image(img_base64)
            gt_key = 'gt-%09d' % index
            gt = str(e.get(gt_key))
        return img, gt
