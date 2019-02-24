# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import ipdb

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, valid=False, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if valid:
            # ipdb.set_trace()
            net_output = model(**sample['net_input'])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }

            return loss, sample_size, logging_output

        # comda method
        elif 'proto_net_input' in sample:
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
                    ).view(-1, 1)
            dec_loss_mask = torch.cat(
                         [sample['orign_dec_loss_mask'], sample['proto_dec_loss_mask']],
                         dim=0
                     ).view(-1, 1)  # [2 x N X L, 1]
            non_pad_mask = concat_target.ne(self.padding_idx)
            # nll_loss = -lprobs.gather(dim=-1, index=concat_target)[non_pad_mask]
            nll_loss = -lprobs.gather(dim=-1, index=concat_target)
            nll_loss = nll_loss.view(-1) * dec_loss_mask.view(-1)
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
            if reduce:
                nll_loss = nll_loss.sum()
                nll_loss = nll_loss / 2.0
                smooth_loss = smooth_loss.sum()
                smooth_loss = smooth_loss / 2.0
            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

            # ipdb.set_trace()
            sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
            return loss, sample_size, logging_output
        else:
            print('Exit')
            ipdb.set_trace()

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
