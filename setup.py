#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:37:26 2020

@author: cjburke
Create the cython binaries for rubik_opt in local directory
python setup.py build_ext --inplace
"""

import distutils.core
import Cython.Build
import numpy

distutils.core.setup(
        ext_modules = Cython.Build.cythonize("*.pyx"), include_dirs=[numpy.get_include()])


