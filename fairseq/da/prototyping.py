"""
Author: Guanlin Li
Date  : Jan. 7 2019
"""

import ipdb

import torch


class Prototyping(object):

    def __init__(self, task, freq=None):
        self.src_dict = task.src_dict
        self.tgt_dict = task.tgt_dict
        self.proto_src_dict = task.proto_src_dict
        self.proto_tgt_dict = task.proto_tgt_dict
        self.mixed_src_dict = task.mixed_src_dict
        self.mixed_tgt_dict = task.mixed_tgt_dict

        if freq is not None:
            self.freq_threshold = freq
        else:
            self.freq_threshold = 0

    def augment(self, sample):
        src_tokens = sample['net_input']['src_tokens']  # [N, L]
        tgt_tokens = sample['target']
        proto_src_tokens = sample['proto_source']
        proto_tgt_tokens = sample['proto_target']
        alignment = sample['alignment']  # List of N alignment_map dicts

        ipdb.set_trace()
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
        ipdb.set_trace()
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

        ipdb.set_trace()

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