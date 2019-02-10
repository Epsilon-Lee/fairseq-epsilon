# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Train a network across multiple GPUs.
"""

from collections import OrderedDict
from itertools import chain
import ipdb

import torch

from fairseq import distributed_utils, models, optim, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler
from fairseq.da import Switchout, Prototyping


class Analyzer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, dummy_batch, oom_batch=None):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.args = args
        self.task = task

        # copy model and criterion to current device
        self.criterion = criterion.cuda()
        if args.fp16:
            self._model = model.half().cuda()
        else:
            self._model = model.cuda()

        self._dummy_batch = dummy_batch
        self._oom_batch = oom_batch
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._wrapped_model = None

        self.init_meters(args)

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters['oom'] = AverageMeter()  # out of memory

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        # build the optimizer when first called
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    def load_checkpoint(self, filename):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = utils.load_model_state(
            filename, self.get_model(),
        )
        return extra_state

    def analyze_step(self, samples, dummy_batch=False):
        """Do forward to get model inner states for analyze"""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # ipdb.set_trace()
        logging_output, sample_size, ooms = {}, 0, 0
        # since args.update_freq is set to 1,
        # the following loop only take 1 cycle
        for i, sample in enumerate(samples):
            # move tensor to GPU
            sample = self._prepare_sample(sample)

            ignore_grad = True
            try:
                # forward and backward
                sample_size, logging_output = self.task.analyze_step(
                    sample, self.model, self.criterion, ignore_grad
                )

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    ooms += 1
                else:
                    raise e

        self.meters['oom'].update(ooms, len(samples))
        if ooms:
            print('| WARNING: OOM detected')
            return {}

        return sample_size, logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def handle_ooms(self, number_of_ooms):
        """
        c10d accumulates/syncs gradients between gpus during backward pass.
        In case of OOMs, gpus may fail to sync, so we manually iterate
        extra to make sure each gpu makes same number of iterations.
        """
        for _ in range(number_of_ooms):
            self.train_step([self._oom_batch], True)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        return utils.move_to_cuda(sample)
