# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Record the encoding representation divergence;
and the decoding predictive (probabilistic)
divergence of each non-generealized token.
"""

import ipdb

import math
import torch.nn.functional as F

import torch

from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('perturb_divergence')
class PerturbDivergence(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.coef = args.coefficient

    def forward(self, model, sample, valid=False, reduce=True):
        """Compute the perturbed representation and predictive probabilities for the given sample.
        """
        # ipdb.set_trace()
        orign_net_input = sample['net_input']
        perturbed_net_input = sample['perturbed_net_input']
        src_tokens = orign_net_input['src_tokens']
        src_lengths = orign_net_input['src_lengths']
        perturbed_src_tokens = perturbed_net_input['src_tokens']
        N = src_tokens.shape[0]

        model.eval()
        # encoder output
        orign_enc_output = model.encoder(src_tokens, src_lengths)  # 1 x L x D
        perturbed_enc_output = model.encoder(perturbed_src_tokens, src_lengths)

        # decoder forward
        orign_decoder_out = model.decoder(
                orign_net_input['prev_output_tokens'],
                orign_enc_output
            )
        perturbed_decoder_out = model.decoder(
                perturbed_net_input['prev_output_tokens'],
                perturbed_enc_output
            )

        # get log probs
        orign_lprobs = model.get_normalized_probs(orign_decoder_out, log_probs=True)
        orign_lprobs = orign_lprobs.view(-1, orign_lprobs.size(-1))
        perturbed_lprobs = model.get_normalized_probs(perturbed_decoder_out, log_probs=True)
        perturbed_lprobs = perturbed_lprobs.view(-1, perturbed_lprobs.size(-1))

        target = sample['target']
        perturbed_target = sample['perturbed_target']
        target = target.view(-1)
        perturbed_target = perturbed_target.view(-1)

        orign_target_lprobs = - F.nll_loss(orign_lprobs, target,
                reduction='none')
        perturbed_target_lprobs = - F.nll_loss(perturbed_lprobs, perturbed_target,
                reduction='none')

        lprob_diverge = perturbed_target_lprobs - orign_target_lprobs
        prob_diverge = torch.exp(perturbed_target_lprobs) - torch.exp(orign_target_lprobs)  # [N x L]
        lprob_diverge = lprob_diverge.view(N, -1)  # [N, L]
        prob_diverge = prob_diverge.view(N, -1)
        # compute encoder representation invariance (enc_inv) loss
        assert 'blockwise_out' in orign_enc_output, 'In perturb analysis'\
                ' mode, transformer encoder forward should pack blockwise output'

        orign_blockwise_out = orign_enc_output['blockwise_out']
        perturbed_blockwise_out = perturbed_enc_output['blockwise_out']
        blockwise_divergence = {}
        for i in range(1, self.args.encoder_layers + 1):
            diverge = perturbed_blockwise_out['block_{}'.format(i)] - orign_blockwise_out['block_{}'.format(i)]
            sq = diverge * diverge  # [L, N, D]
            sq_diverge = torch.sum(sq, 2)  # [L, N]
            blockwise_divergence['block_{}'.format(i)] = sq_diverge.t()  # [N, L]

        # logging statistics
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'blockwise_diverge': blockwise_divergence,
            'lprob_diverge': lprob_diverge,
            'prob_diverge': prob_diverge,
        }
        return sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        agg_output = {}
        return agg_output
