"""
Author: Guanlin Li
Date  : Jan. 7 2019
"""

import ipdb
import random
import torch

from ..tasks.translation_comda_xxx import ComdaXXXTranslationTask


class Prototyping(object):

    def __init__(self, task, freq=None):
        self.src_dict = task.src_dict
        self.tgt_dict = task.tgt_dict
        self.left_pad_source = task.args.left_pad_source
        self.left_pad_target = task.args.left_pad_target
        if not isinstance(task, ComdaXXXTranslationTask):
            self.proto_src_dict = task.proto_src_dict
            self.proto_tgt_dict = task.proto_tgt_dict
            self.mixed_src_dict = task.mixed_src_dict
            self.mixed_tgt_dict = task.mixed_tgt_dict
            self.prototyping_strategy = 'length-invariant'
        else:
            self.prototyping_strategy = 'length-variant'

        if freq is not None:
            self.freq_threshold = freq
        else:
            self.freq_threshold = 0

    def augment(self, sample, dummy_batch=False):
        src_tokens = sample['net_input']['src_tokens']  # [N, L]
        tgt_tokens = sample['target']
        if self.prototyping_strategy == 'length-variant':
            new_sample = self._augment_length_variant(
                    sample,
                    dummy_batch=dummy_batch)
            return new_sample
        proto_src_tokens = sample['proto_source']
        proto_tgt_tokens = sample['proto_target']
        alignment = sample['alignment']  # List of N alignment_map dicts

        N = src_tokens.shape[0]

        # unk based prototyping
        for i in range(N):
            new_src, new_tgt = self._unk_transform(
                src_tokens[i], proto_src_tokens[i],
                tgt_tokens[i], proto_tgt_tokens[i],
                alignment[i]
            )
            src_tokens[i] = new_src
            tgt_tokens[i] = new_tgt

        # freq based prototyping
        for i in range(N):
            new_src, new_tgt = self._token_freq_transform(
                src_tokens[i], proto_src_tokens[i],
                tgt_tokens[i], proto_tgt_tokens[i],
                alignment[i]
            )
            src_tokens[i] = new_src
            tgt_tokens[i] = new_tgt

        # TODO: uncertainty based prototyping
        for i in range(N):
            new_src, new_tgt = self._uncertainty_transform(
                src_tokens[i], proto_src_tokens[i],
                tgt_tokens[i], proto_tgt_tokens[i],
                alignment[i]
            )
            src_tokens[i] = new_src
            tgt_tokens[i] = new_tgt

        sample['net_input']['src_tokens'] = src_tokens
        sample['target'] = tgt_tokens

        return sample

    def _unk_transform(self, src, proto_src,
                   tgt, proto_tgt, alignment_map):
        # ipdb.set_trace()
        src, tgt = src.tolist(), tgt.tolist()
        proto_src, proto_tgt = proto_src.tolist(), proto_tgt.tolist()
        tgt_len = len(tgt)
        tgt_js = range(1, tgt_len + 1)
        for j, idx in zip(tgt_js, tgt):
            # break when reach eos idx
            if idx == self.tgt_dict.eos():
                break
            # try substitute when encounter unk idx
            if idx == self.tgt_dict.unk():
                if j in alignment_map:
                    i = alignment_map[j]
                    # get target proto symbol, and substitute
                    tgt_proto_symbol = self.proto_tgt_dict.symbols[proto_tgt[j]]
                    tgt[j] = self.mixed_tgt_dict.indices[tgt_proto_symbol]
                    # get source proto symbol, and substitute
                    src_proto_symbol = self.proto_src_dict.symbols[proto_src[i]]
                    src[i] = self.mixed_src_dict.indices[src_proto_symbol]

        # create inverse alignment map
        inverse_alignment_map = {}
        for j, i in alignment_map.items():
            inverse_alignment_map[i] = j
        src_len = len(src)
        src_is = range(1, src_len + 1)
        for i, idx in zip(src_is, src):
            # break when reach eos idx
            if idx == self.src_dict.eos():
                break
            if idx == self.src_dict.unk():
                if i in inverse_alignment_map:
                    j = inverse_alignment_map[i]
                    # get source idx, and substitute
                    src_proto_symbol = self.proto_src_dict.symbols[proto_src[i]]
                    src[i] = self.mixed_src_dict.indices[src_proto_symbol]
                    # get target idx, and substitute
                    tgt_proto_symbol = self.proto_tgt_dict.symbols[proto_tgt[j]]
                    tgt[j] = self.mixed_tgt_dict[tgt_proto_symbol]

        new_src = torch.LongTensor(src)
        new_tgt = torch.LongTensor(tgt)

        # ipdb.set_trace()

        return new_src, new_tgt

    def _token_freq_transform(self, src, proto_src,
                             tgt, proto_tgt, alignment_map):
        """
            Currently, the substitution strategy is: given a freq threshold `freq`,
            for token that in the original dict, if its count < `freq`, substitute
            the token to its proto token.
        """
        eos, pad, unk = self.src_dict.eos(), self.src_dict.pad(), self.src_dict.unk()

        # tgt_j
        # for

        new_src = torch.LongTensor(src)
        new_tgt = torch.LongTensor(tgt)

        return new_src, new_tgt

    def _uncertainty_transform(self, src, proto_src,
                              tgt, proto_tgt, alignment_map):
        """
            pass
        """

        new_src = torch.LongTensor(src)
        new_tgt = torch.LongTensor(tgt)

        return new_src, new_tgt

    def _augment_length_variant(self, sample, dummy_batch=False):

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if True and not dummy_batch:
            # if True and not dummy_batch:
                eos_idx = self.tgt_dict.eos()
                # assert src[-1] == eos_idx
                length = dst.shape[0]
                pos = length
                while src[pos - 1] != eos_idx:
                    pos -= 1
                    if pos == 0:
                        print('AssertionError: cannot find pos equal eos_idx')
                        ipdb.set_trace()
                # ipdb.set_trace()
                dst[0] = eos_idx
                dst[1:pos] = src[0:pos - 1]
            else:
                dst.copy_(src)

        # ipdb.set_trace()
        net_input = sample['net_input']
        src_tokens = net_input['src_tokens']
        src_lengths = net_input['src_lengths']
        tgt_tokens = sample['target']
        # tgt_lengths = sample['tgt_lengths']
        alignment = sample['alignment']
        proto_src_tokens = src_tokens.clone()
        proto_src_lengths = src_lengths.clone()
        proto_tgt_tokens = tgt_tokens.clone()
        proto_prev_output_tokens = tgt_tokens.clone()
        enc_feat_mask_list = []
        for src, src_len, tgt, prev_tgt, align in zip(
                proto_src_tokens, proto_src_lengths,
                proto_tgt_tokens,
                proto_prev_output_tokens, alignment):

            num_aligned_phrases = len(align)
            # randomly picked phrase pair: 'm-n:p-q'
            randn_ppair = align[random.randint(0, num_aligned_phrases - 1)]
            m_n, p_q = randn_ppair.split(':')
            m, n = m_n.split('-')
            p, q = p_q.split('-')
            m, n, p, q = int(m), int(n), int(p), int(q)
            if self.left_pad_source:
                src_pad_len = src_tokens.shape[1] - src_len
            else:
                src_pad_len = 0
            if self.left_pad_target:
                raise ValueError('left_pad_target should be False!')
                # tgt_pad_len = tgt_tokens.shape[1] - tgt_len
            else:
                tgt_pad_len = 0
            # src[m : n + 1] = self.src_dict.xxx()
            src[src_pad_len + m : src_pad_len + n + 1] = self.src_dict.xxx()
            tgt[tgt_pad_len + p : tgt_pad_len + q + 1] = self.tgt_dict.xxx()
            enc_feat_mask = 1 - src.eq(self.src_dict.xxx()) - src.eq(self.src_dict.pad())
            enc_feat_mask_list.append(enc_feat_mask.tolist())
            copy_tensor(tgt, prev_tgt)
        # ipdb.set_trace()
        enc_feat_mask = torch.Tensor(enc_feat_mask_list)
        sample['enc_feat_mask'] = enc_feat_mask
        sample['proto_net_input'] = {}
        sample['proto_net_input']['src_tokens'] = proto_src_tokens
        sample['proto_net_input']['src_lengths'] = proto_src_lengths
        sample['proto_net_input']['prev_output_tokens'] = proto_prev_output_tokens

        sample['proto_target'] = proto_tgt_tokens

        return sample



