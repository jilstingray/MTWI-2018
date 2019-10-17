# -*- coding: utf-8 -*-

"""Fast R-CNN non-maximum suppression Cython setup

run command: python setup_cython.py build_ext --inplace
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize(Extension(
    'nms',
    sources=['nms.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))
