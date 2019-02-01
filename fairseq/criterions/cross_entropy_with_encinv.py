# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Cross entropy loss with encoding invariance regularization:
    1. First do model.encoder.forward() on concated instance and proto
    instance;
    2. Then compute encoding invariance loss with masked source positions;
    3. Then call model.decoder.forward() to get logits;
    4. Finally add enc_inv loss and xent loss with $\lambda$ as interpolation
    coefficient.
"""

import ipdb

import math
import torch.nn.functional as F

import torch

from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy_with_encinv')
class CrossEntropyCriterionWithEncInv(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.coef = args.coefficient

    def forward(self, model, sample, valid=False, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # if not using compositional gradient
        if valid or self.args.no_comgrad:
            # do simple forward computation
            net_output = model(**sample['net_input'])
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output).view(-1)
            loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                              reduce=reduce)
            sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
            return loss, sample_size, logging_output

        # ipdb.set_trace()
        # src_tokens, src_lengths, prev_output_tokens
        orign_net_input = sample['net_input']
        proto_net_input = sample['proto_net_input']

        N = orign_net_input['src_tokens'].shape[0]

        concat_src_tokens = torch.cat(
                [orign_net_input['src_tokens'],
                proto_net_input['src_tokens']], 0)  # [2N, L]
        concat_src_lengths = torch.cat(
                [orign_net_input['src_lengths'],
                proto_net_input['src_lengths']], 0)  # [2N, L]
        concat_prev_output_tokens = torch.cat(
                [orign_net_input['prev_output_tokens'],
                proto_net_input['prev_output_tokens']], 0)  # [2N, L]
        # encoder output
        enc_output = model.encoder(concat_src_tokens, concat_src_lengths)
        encoder_out = enc_output['encoder_out']  # [L, 2N, D]
        encoder_padding_mask = enc_output['encoder_padding_mask']  # [2N, L]
        # slice to get orign encoder output and proto encoder output
        orign_encoder_out, proto_encoder_out = encoder_out[:, 0:N, :], \
                encoder_out[:, N:2*N, :]
        # orign_encoder_padding_mask = encoder_padding_mask[0:N]  # [N, L]
        # orign_enc_output = {
        #         'encoder_out': orign_encoder_output,
        #         'encoder_padding_mask': orign_encoder_padding_mask
        #     }

        # ipdb.set_trace()
        # decoder forward
        decoder_out = model.decoder(
                concat_prev_output_tokens,
                enc_output
            )

        # get log probs
        lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = model.get_targets(sample, decoder_out).view(-1)
        concat_target = torch.cat(
                    [sample['target'], sample['proto_target']],
                    0
                ).view(-1)
        # divide 2: to get averaged one batch total loss
        loss = F.nll_loss(
                lprobs, concat_target, size_average=False,
                ignore_index=self.padding_idx,
                reduce=reduce) / 2.0

        # compute encoder representation invariance (enc_inv) loss
        delta = orign_encoder_out - proto_encoder_out
        delta_square_sum = torch.sum(delta * delta, 2).t()  # B x T
        # print('enc_feat_mask.shape:')
        # print(sample['enc_feat_mask'].shape)
        # print('delta_square_sum.shape:')
        # print(delta_square_sum.shape)
        # print()
        masked_delta_l2 = sample['enc_feat_mask'] * delta_square_sum  # B x T
        # avg_masked_delta_l2 = masked_delta_l2 / torch.sum(sample['enc_feat_mask'])
        masked_delta_l2 = torch.sum(masked_delta_l2)
        scaled_masked_delta_l2 = (sample['ntokens'] / N) * masked_delta_l2

        # compute final loss through interpolation
        # ipdb.set_trace()
        coef = self.coef
        # final_loss = coef * loss + (1 - coef) * masked_delta_l2
        final_loss = coef * loss + (1 - coef) * scaled_masked_delta_l2

        # logging statistics
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'enc_inv_loss': scaled_masked_delta_l2.data,
        }
        if self.args.repeat_batch > 1:
            final_loss *= self.args.repeat_batch
        return final_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        enc_inv_loss_sum = sum(log.get('enc_inv_loss', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'enc_inv_loss': enc_inv_loss_sum / nsentences,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
