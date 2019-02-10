# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from fairseq import options, utils
from fairseq.data import (
    data_utils, Dictionary, PerturbAnalysisDataset, ConcatDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset
)

from . import FairseqTask, register_task


@register_task('perturb_analysis')
class PerturbAnalysisTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--use-xxx', action='store_true',
                            help='use abstract token <xxx>, only true for'\
                            'comda')
        parser.add_argument('--coefficient', default=0.5, type=float,
                            help='the interpolation coefficient before'\
                            'enc_inv loss and mle loss')
        parser.add_argument("--length-invariant", default='True', type=str,
                            help='whether to use same number of placeholder'\
                            'xxx to replace each specifit token or to use a'\
                            'sinple placeholder')
        parser.add_argument("--repeat-batch", default=1, type=int,
                            help='scale factor of gradient update, to emulate'\
                            'repeated batches')
        parser.add_argument("--no-comgrad", action='store_true',
                            help='use offline augmented data instead of'\
                            'compositional batching curriculum')
        parser.add_argument("--is-baseline", action='store_true',
                            help='flag to annotate whether analyzed model is'\
                            'baseline')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = Dictionary.load(os.path.join(args.data[0],
            'dict.{}.txt'.format(args.source_lang)), use_xxx=args.use_xxx)
        tgt_dict = Dictionary.load(os.path.join(args.data[0],
            'dict.{}.txt'.format(args.target_lang)), use_xxx=args.use_xxx)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        if args.use_xxx:
            assert src_dict.xxx() == tgt_dict.xxx()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, **kwargs):
        """load a datasets for analysis
        """
        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        data_path = self.args.data[0]
        src, tgt = self.args.source_lang, self.args.target_lang
        prefix = os.path.join(data_path, '{}.{}-{}.'.format('origin', src, tgt))
        src_dataset = indexed_dataset(prefix + src, self.src_dict)
        tgt_dataset = indexed_dataset(prefix + tgt, self.tgt_dict)
        print('| {} {} {} examples'.format(data_path, prefix + '.src', len(src_dataset)))
        print('| {} {} {} examples'.format(data_path, prefix + '.tgt', len(src_dataset)))

        prefix = os.path.join(data_path, '{}.{}-{}.'.format('perturb', src, tgt))
        perturbed_src_dataset = indexed_dataset(prefix + src, self.src_dict)
        perturbed_tgt_dataset = indexed_dataset(prefix + tgt, self.tgt_dict)
        print('| {} {} {} examples'.format(data_path, prefix + '.src', len(src_dataset)))
        print('| {} {} {} examples'.format(data_path, prefix + '.tgt', len(src_dataset)))

        self.datasets['perturb'] = PerturbAnalysisDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            perturbed_src_dataset, perturbed_tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
