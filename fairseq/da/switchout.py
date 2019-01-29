"""
Author: Guanlin Li
Date  : Jan. 7 2019
"""

import ipdb

import torch


class Switchout(object):

    def __init__(self, task):
        self.tau = {}
        self.tau['src'] = 0.95
        self.tau['tgt'] = 0.90

        self.vocab_size = {}
        self.vocab_size['src'] = len(task.src_dict)
        self.vocab_size['tgt'] = len(task.tgt_dict)
        self.pad_index = task.src_dict.pad()
        self.eos_index = task.src_dict.eos()
        self.start_idx = task.src_dict.nspecial

        # debug
        self.src_dict = task.src_dict
        self.tgt_dict = task.tgt_dict

    def augment(self, sample, dummy_batch=False):
        """
        Refer to the appendix of the original paper for reference
        implementation.

        :param sample:
        :param task:
        :return:
        """
        src_tokens = sample['net_input']['src_tokens']
        tgt_tokens = sample['target']
        sample['net_input']['src_tokens'] = self._transform(src_tokens, 'src')
        sample['target'] = self._transform(tgt_tokens, 'tgt')

        return sample

    def _transform(self, sents, lang='src'):

        mask = torch.eq(sents, self.pad_index) | torch.eq(sents, self.eos_index)
        unmask = (1 - mask).float()
        lengths = unmask.sum(dim=1)
        batch_size, n_steps = sents.size()

        # first, sample the number of words to corrupt for each sentence
        logits = torch.arange(n_steps, dtype=torch.float32)
        logits = logits.mul_(-1).unsqueeze(0).expand_as(
            sents).contiguous().masked_fill_(mask, -float("inf"))
        probs = torch.nn.functional.softmax(logits.mul_(self.tau[lang]), dim=1)
        num_words = torch.distributions.Categorical(probs).sample()  # how many words to switch out
        # if num_words[0] > 0:
        #     ipdb.set_trace()

        # sample the corrupted positions
        corrupt_pos = num_words.data.float().div_(lengths).unsqueeze(
            1).expand_as(sents).contiguous().masked_fill_(mask, 0)
        corrupt_pos = torch.bernoulli(corrupt_pos).byte()
        total_words = int(corrupt_pos.sum())

        # sample the corrupted values, which will be added to sents
        corrupt_val = torch.LongTensor(total_words)
        corrupt_val = corrupt_val.random_(self.start_idx, self.vocab_size[lang])
        corrupts = torch.zeros(batch_size, n_steps).long()
        corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)
        sampled_sents = sents.add(corrupts).remainder_(self.vocab_size[lang])

        return sampled_sents
