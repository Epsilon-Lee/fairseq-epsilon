"""
Author: Guanlin Li
Date  : Jan. 7 2019
"""

import ipdb
import random
import torch

from ..tasks.translation_comda_xxx import ComdaXXXTranslationTask
from ..data import data_utils, FairseqDataset


class Prototyping(object):

    def __init__(self, task, freq=None):
        self.args = task.args
        self.src_dict = task.src_dict
        self.tgt_dict = task.tgt_dict
        self.left_pad_source = task.args.left_pad_source
        self.left_pad_target = task.args.left_pad_target
        if not isinstance(task, ComdaXXXTranslationTask):
            self.proto_src_dict = task.proto_src_dict
            self.proto_tgt_dict = task.proto_tgt_dict
            self.mixed_src_dict = task.mixed_src_dict
            self.mixed_tgt_dict = task.mixed_tgt_dict
            self.prototyping_strategy = 'multiple-proto-tokens'
        else:
            if task.args.length_invariant == 'True':
                self.prototyping_strategy = 'xxx-length-invariant'
            else:
                self.prototyping_strategy = 'xxx-length-variant'

        if freq is not None:
            self.freq_threshold = freq
        else:
            self.freq_threshold = 0

    def augment(self, sample, dummy_batch=False):

        if self.prototyping_strategy == 'xxx-length-invariant':
            new_sample = self._augment_xxx_length_invariant(
                    sample,
                    dummy_batch=dummy_batch)
            return new_sample
        elif self.prototyping_strategy == 'xxx-length-variant':
            new_sample = self._augment_xxx_length_variant(
                    sample,
                    dummy_batch=dummy_batch)
            return new_sample

        src_tokens = sample['net_input']['src_tokens']  # [N, L]
        tgt_tokens = sample['target']
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

    def _augment_xxx_length_invariant(self, sample, dummy_batch=False):

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
        prev_output_tokens = net_input['prev_output_tokens']
        tgt_tokens = sample['target']
        # tgt_lengths = sample['tgt_lengths']
        alignment = sample['alignment']
        proto_src_tokens = src_tokens.clone()
        proto_src_lengths = src_lengths.clone()
        proto_prev_output_tokens = prev_output_tokens.clone()
        proto_tgt_tokens = tgt_tokens.clone()
        enc_feat_mask_list = []
        dec_loss_mask_list = []
        N, tgt_L = tgt_tokens.shape
        for src, src_len, tgt, prev_tgt, align in zip(
                proto_src_tokens, proto_src_lengths,
                proto_tgt_tokens,
                proto_prev_output_tokens, alignment):

            num_aligned_phrases = len(align)
            # if len(align) == 0, reserve the original sentence pair
            if num_aligned_phrases == 0:
                enc_feat_mask = 1 - src.eq(self.src_dict.pad())
                dec_loss_mask = torch.zeros([tgt_L])
            else:
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
                copy_tensor(tgt, prev_tgt)
                enc_feat_mask = 1 - src.eq(self.src_dict.xxx()) - src.eq(self.src_dict.pad())
                dec_loss_mask = 1 - tgt.eq(self.tgt_dict.pad())

            enc_feat_mask_list.append(enc_feat_mask.tolist())
            dec_loss_mask_list.append(dec_loss_mask.tolist())

        enc_feat_mask = torch.Tensor(enc_feat_mask_list)
        dec_loss_mask = torch.Tensor(dec_loss_mask_list)
        sample['enc_feat_mask'] = enc_feat_mask
        orign_dec_loss_mask = 1.0 - tgt_tokens.eq(self.tgt_dict.pad())
        sample['orign_dec_loss_mask'] = orign_dec_loss_mask.float()
        sample['proto_dec_loss_mask'] = dec_loss_mask.float()
        sample['proto_net_input'] = {}
        sample['proto_net_input']['src_tokens'] = proto_src_tokens
        sample['proto_net_input']['src_lengths'] = proto_src_lengths
        sample['proto_net_input']['prev_output_tokens'] = proto_prev_output_tokens
        sample['proto_target'] = proto_tgt_tokens

        return sample

    def _augment_xxx_length_variant(self, sample, dummy_batch=False):
        """
        Need self.left_pad_source == True and self.left_pad_target == False
        """

        # assert
        assert self.left_pad_source == True, 'self.left_pad_source should be True'
        assert self.left_pad_target == False, 'self.left_pad_target should be False'

        def merge(key, left_pad, move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            )

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

        net_input = sample['net_input']
        src_tokens = net_input['src_tokens']
        src_lengths = net_input['src_lengths']
        N, src_max_len = src_tokens.shape[0], src_tokens.shape[1]
        tgt_tokens = sample['target']
        tgt_max_len = tgt_tokens.shape[1]
        alignment = sample['alignment']

        # new source
        proto_src_tokens = src_tokens.clone().tolist()  # [N, L]
        new_proto_src_tokens = []
        proto_src_lengths = []
        # new target
        proto_tgt_tokens = tgt_tokens.clone().tolist()
        new_proto_tgt_tokens = []
        proto_tgt_lengths = []

        proto_prev_output_tokens = []
        enc_feat_mask_list = []

        for src, src_len, tgt, align in zip(
                proto_src_tokens,
                src_lengths,
                proto_tgt_tokens,
                alignment
            ):

            new_src = []
            new_tgt = []

            num_aligned_phrases = len(align)
            # randomly picked phrase pair: 'm-n:p-q'
            randn_ppair = align[random.randint(0, num_aligned_phrases - 1)]
            m_n, p_q = randn_ppair.split(':')
            m, n = m_n.split('-')
            p, q = p_q.split('-')
            m, n, p, q = int(m), int(n), int(p), int(q)

            src_pad_len = src_max_len - src_len
            new_src.extend(src[src_pad_len : src_pad_len + m])
            new_src.append(self.src_dict.xxx())
            new_src.extend(src[src_pad_len + n + 1 : ])
            new_proto_src_tokens.append(new_src)
            # src[m : n + 1] = self.src_dict.xxx()

            tgt_len = tgt.index(self.tgt_dict.eos()) + 1
            new_tgt.extend(tgt[0 : p])
            new_tgt.append(self.tgt_dict.xxx())
            new_tgt.extend(tgt[q + 1 : tgt_len])
            new_proto_tgt_tokens.append(new_tgt)

            proto_src_lengths.append(len(new_src))
            proto_tgt_lengths.append(len(new_tgt))

        # build proto src/tgt tensor with same max src/tgt length
        proto_src_tokens = torch.LongTensor(
                N, src_max_len).fill_(self.src_dict.pad())
        proto_tgt_tokens = torch.LongTensor(
                N, tgt_max_len).fill_(self.tgt_dict.pad())
        proto_prev_output_tokens = torch.LongTensor(N,
            tgt_max_len).fill_(self.tgt_dict.pad())
        for proto_src, src, src_len, proto_tgt, tgt, tgt_len, proto_prev_tgt in zip(
                proto_src_tokens,
                new_proto_src_tokens,
                proto_src_lengths,
                proto_tgt_tokens,
                new_proto_tgt_tokens,
                proto_tgt_lengths,
                proto_prev_output_tokens
            ):
            src_pad_len = src_max_len - src_len
            proto_src[src_pad_len : ] = torch.LongTensor(src)
            proto_tgt[0 : tgt_len] = torch.LongTensor(tgt)

            # encoder feature mask
            enc_feat_mask = 1.0 - torch.eq(proto_src, self.src_dict.pad()) - torch.eq(proto_src, self.src_dict.xxx())
            enc_feat_mask_list.append(enc_feat_mask.tolist())

            # proto_prev_output_tokens
            # import ipdb; ipdb.set_trace()
            copy_tensor(proto_tgt, proto_prev_tgt)

        proto_src_lengths = torch.LongTensor(proto_src_lengths)

        enc_feat_mask = torch.Tensor(enc_feat_mask_list)
        sample['enc_feat_mask'] = enc_feat_mask
        sample['proto_net_input'] = {}
        sample['proto_net_input']['src_tokens'] = proto_src_tokens
        sample['proto_net_input']['src_lengths'] = proto_src_lengths
        sample['proto_net_input']['prev_output_tokens'] = proto_prev_output_tokens
        sample['proto_target'] = proto_tgt_tokens
        orign_target_mask = 1.0 - torch.eq(sample['target'], self.tgt_dict.pad()).float()
        proto_target_mask = 1.0 - \
                torch.eq(proto_tgt_tokens, self.tgt_dict.pad()).float() - \
                torch.eq(proto_tgt_tokens, self.tgt_dict.xxx()).float() * (1.0 - \
                        self.args.xxx_loss_coefficient)
        # import ipdb; ipdb.set_trace()
        sample['orign_dec_loss_mask'] = orign_target_mask
        sample['proto_dec_loss_mask'] = proto_target_mask

        return sample
