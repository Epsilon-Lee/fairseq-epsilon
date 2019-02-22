# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os
import torch
import ipdb

from fairseq import options, utils
from fairseq.data import (
    data_utils, Dictionary, ConcatDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset,
    IndexedPhraseAlignmentDataset, LanguagePairDataset,
    ComdaXXXTranslationDataset
)

from . import FairseqTask, register_task


@register_task('translation_comda_xxx')
class ComdaXXXTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    A new translation class with COMpositional Data Augmentation (Comda).

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
        parser.add_argument("--phrase-length", type=int, default=0,
                            help='if > 0, use fixed length phrase pair align')
        parser.add_argument('--xxx-loss-coefficient', default=1.0, type=float,
                            help='the xxx token has the loss interpolated'\
                            'with other tokens, thus reweight its gradient')
        # fmt: on

    @staticmethod
    def load_pretrained_model(path, src_dict_path, tgt_dict_path,
                              arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        src_dict = Dictionary.load(src_dict_path)
        tgt_dict = Dictionary.load(tgt_dict_path)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        task = ComdaTranslationTask(
            args, src_dict, tgt_dict,
        )
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

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
        assert src_dict.xxx() == tgt_dict.xxx()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        def indexed_alignment_dataset(path):
            if IndexedAlignmentDataset.exists(path):
                return IndexedAlignmentDataset(path)
            else:
                ValueError("Alignment file path must be given!")

        def indexed_phrase_alignment_dataset(path):
            if IndexedPhraseAlignmentDataset.exists(path):
                return IndexedPhraseAlignmentDataset(path)
            else:
                ValueError("Alignment file path must be given!")

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                if split == 'train' and self.args.debug:
                    src_datasets.append(indexed_dataset(prefix + src + '.debug', self.src_dict))
                    tgt_datasets.append(indexed_dataset(prefix + tgt + '.debug', self.tgt_dict))
                else:
                    src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                    tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        if split == 'train' and not self.args.no_comgrad:
            # ipdb.set_trace()
            if self.args.phrase_length > 0:
                align_path = prefix + "phrase-align.length%d" % self.args.phrase_length
            else:
                align_path = prefix + "phrase-align"
            if self.args.debug and self.args.phrase_length == 0:
                align_path += '.debug'
            phrase_alignment_dataset = indexed_phrase_alignment_dataset(align_path)
        else:
            if split == 'train' and self.args.no_comgrad:
                print("No compositional gradient is True during training")
            else:
                print("WARNING: valid alignment should be prepared in the data folder.")
            phrase_alignment_dataset = None

        if split == 'train' and not self.args.no_comgrad:
            self.datasets[split] = ComdaXXXTranslationDataset(
                src_dataset, src_dataset.sizes, self.src_dict,
                tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                phrase_alignment_dataset,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
        else:
            self.datasets[split] = LanguagePairDataset(
                src_dataset, src_dataset.sizes, self.src_dict,
                tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
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

    def valid_step(self, sample, model, criterion):
        """Rewrite FairseqTask's same name method."""
        # ipdb.set_trace()
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(
                    model, sample, valid=True)
        return loss, sample_size, logging_output
