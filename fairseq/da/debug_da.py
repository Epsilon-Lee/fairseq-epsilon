"""
Author: Guanlin Li
Date  : Jan. 7 2019
"""

import ipdb
import random
import torch

from ..tasks.translation_comda_xxx import ComdaXXXTranslationTask
from ..data import data_utils, FairseqDataset


class Debug_DA(object):

    def __init__(self, task, freq=None):
        self.args = task.args
        self.src_dict = task.src_dict
        self.tgt_dict = task.tgt_dict
        self.left_pad_source = task.args.left_pad_source
        self.left_pad_target = task.args.left_pad_target

    def augment(self, sample, dummy_batch=False):

        if 'net_input' in sample:
            print('sample has no key net_input')
            net_input = sample['net_input']

        return sample

