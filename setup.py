#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


setup(
    name="ibl_tools",
    version="0.1.0",
    description="Analysis of Behavior Data for IBL",
    author="E. Kelly Buchanan",
    author_email="ekellbuchanan@gmail.com",
    url="https://github.com/ekellbuch/ibl_tools",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[
        splitext(basename(path))[0]
        for path in glob("src/ibl_tools/*.py", recursive=True)
    ],
    include_package_data=True,
    zip_safe=False,
)
