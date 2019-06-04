# README

The Python implementation of Connectionist Text Proposal Network (CTPN) targeting at MTWI 2018 Challenge 2.

## Competition Details

https://tianchi.aliyun.com/competition/entrance/231685/introduction

## Environment

* Python 3.x with PyTorch (0.4 or newer)

* cython, configphaser, lmdb, matplotlib, numpy, opencv-python are required.

## Run this files before training

* Run setup_cython.py FIRST: 

```bash
cd lib
python setup_cython.py build_ext --inplace
```

* Modify config.py and use it to generate config file for your environment.

* If you've downloaded the original MTWI_2018 dataset from Aliyun, try to use `dataset_handler.reorganize_dataset()` to reorganize it, since some pictures may not be read as RGB channels.

## Infer an image

After training, open infer.py and set `MODEL` as your trained model's path. Then run infer.py:

```bash
python infer.py [url] [infer_one(random)] [cpu(gpu)]
```

## Reference

https://arxiv.org/abs/1609.03605

https://github.com/AstarLight/Lets_OCR/tree/master/detector/ctpn
