#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:37:26 2020

@author: cjburke
Create the cython binaries for rubik_opt in local directory
python setup.py build_ext --inplace
"""

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
        Extension(
                "rubik_cython_moves",
                ["rubik_cython_moves.pyx"],
                include_dirs=[numpy.get_include()],
                extra_compile_args = ["-O3"]
        ),
        Extension(
                "rubik_cython_moves_12p7",
                ["rubik_cython_moves_12p7.pyx"],
                include_dirs=[numpy.get_include()],
                extra_compile_args = ["-O3"]
        ),
        Extension(
                "rubik_cython_roll_buffdq_solve",
                ["rubik_cython_roll_buffdq_solve.pyx"],
                include_dirs=[numpy.get_include()],
                extra_compile_args = ["-O3"]
        ),
        Extension(
                "rubik_cython_roll_buffdq_solve_MP",
                ["rubik_cython_roll_buffdq_solve_MP.pyx"],
                include_dirs=[numpy.get_include()],
                extra_compile_args = ["-O3"]
        )
]

setup(ext_modules=cythonize(extensions))
