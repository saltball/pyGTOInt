# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : conf.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import os
import logging
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

_ELEMENT_DB_PATH_ = os.path.join(PROJECT_ROOT, r"data/elements.db")

_DEVICE_ = "cuda"

# __EPS__OF__BOYSFUNCTION__ = np.float64(x=1.0e-17)

__MAX__ITER__OF__BOYSFUNC__ = 100
