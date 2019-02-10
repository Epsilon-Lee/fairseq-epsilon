#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Analyze a trained  model.
"""

import collections
import itertools
import os
import math
import random
import ipdb

import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.analyzer import Analyzer
from fairseq.meters import AverageMeter, StopwatchMeter


def main(args):
    if args.distributed_world_size > 1:
        raise ValueError('Do not support multiple gpu analysis.')
    assert args.update_freq == [1], 'update_freq should be set to 1'

    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., analysis etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset(task)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('perturb').get_dummy_batch(args.max_tokens, max_positions)
    oom_batch = task.dataset('perturb').get_dummy_batch(1, max_positions)

    # Build analyzer
    analyzer = Analyzer(args, task, model, criterion, dummy_batch, oom_batch)
    print('| analyzing on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize data loader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset('perturb'),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    # ipdb.set_trace()
    # Load the appointed checkpoint for analyzing, set '--restore-file'
    load_checkpoint(args, analyzer, epoch_itr)

    # Analyze until the end of the input file
    max_epoch = args.max_epoch or math.inf
    analyze_meter = StopwatchMeter()
    analyze_meter.start()
    while epoch_itr.epoch < max_epoch:
        # train for one epoch
        analyze(args, analyzer, task, epoch_itr)

    print('| done analysis in {:.1f} seconds'.format(analyze_meter.sum))


def analyze(args, analyzer, task, epoch_itr):
    """Analyze the model for one epoch."""

    # Initialize data iterator
    update_freq = args.update_freq[0]
    itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus)
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    fn = os.path.join(args.save_dir, 'analysis.txt')
    f = open(fn, 'w')
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        # ipdb.set_trace()
        batch_size = samples[0]['nsentences']
        sample_size, log_output = analyzer.analyze_step(samples)
        if log_output is {}:
            continue
        for j in range(batch_size):
            src = samples[0]['net_input']['src_tokens'][j]
            perturbed_src = samples[0]['perturbed_net_input']['src_tokens'][j]
            tgt = samples[0]['target'][j]
            perturbed_tgt = samples[0]['perturbed_target'][j]
            line_src = task.src_dict.string(src)
            line_tgt = task.tgt_dict.string(tgt)
            line_perturbed_src = task.src_dict.string(perturbed_src)
            line_perturbed_tgt = task.tgt_dict.string(perturbed_tgt)
            src = line_src.split() + ['<eos>']
            tgt = line_tgt.split() + ['<eos>']
            perturbed_src = line_perturbed_src.split() + ['<eos>']
            perturbed_tgt = line_perturbed_tgt.split() + ['<eos>']

            f.write('src          :' + line_src.encode('utf-8') + '\n')
            f.write('perturbed_src:' + line_perturbed_src.encode('utf-8') + '\n')
            f.write('tgt          :' + line_tgt.encode('utf-8') + '\n')
            f.write('perturbed_tgt:' + line_perturbed_tgt.encode('utf-8') + '\n')
            f.write('\n')

            prob_diverge = log_output['prob_diverge'][j]
            lprob_diverge = log_output['lprob_diverge'][j]
            blockwise_diverge = list(log_output['blockwise_diverge']['block_{}'.format(k)][j]
                    for k in range(1, args.encoder_layers + 1))
            line_prob_diverge = ''
            for t, pd in enumerate(prob_diverge.tolist()):
                line_prob_diverge += '%.4f(%s) ' % (pd, tgt[t] + ' ' +
                        perturbed_tgt[t])
            line_prob_diverge = line_prob_diverge[:-1]
            line_lprob_diverge = ''
            for lpd in lprob_diverge.tolist():
                line_lprob_diverge += '%.4f ' % lpd
            line_lprob_diverge = line_lprob_diverge[:-1]
            f.write('prob_diverge   : {}'.format(line_prob_diverge.encode('utf-8')) + '\n')
            f.write('lprob_diverge  : {}'.format(line_lprob_diverge.encode('utf-8')) + '\n')
            for k in range(1, args.encoder_layers + 1):
                line_blockwise_diverge = ''
                for bd in blockwise_diverge[k - 1].tolist():
                    line_blockwise_diverge += '%.4f ' % bd
                line_blockwise_diverge = line_blockwise_diverge[:-1]
                f.write('block_{}_diverge: {}'.format(k,
                    line_blockwise_diverge.encode('utf-8')) + '\n')
            f.write('\n')
            f.write('\n')
        # log mid-epoch stats
        stats = get_analyze_stats(sample_size)
        progress.log(stats)


def get_analyze_stats(sample_size):
    stats = collections.OrderedDict()
    stats['sample_size'] = sample_size
    return stats


def load_checkpoint(args, analyzer, epoch_itr):
    """Load a checkpoint, mainly the model parameters"""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        # load model parameters
        extra_state = analyzer.load_checkpoint(checkpoint_path) 
        return True
    return False


def load_dataset(task):
    task.load_dataset()


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        # Set distributed training parameters for a single node.
        args.distributed_world_size = torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_port = port + 1

        multiprocessing_main(args)
    else:
        main(args)
