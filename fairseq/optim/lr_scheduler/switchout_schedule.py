# Jan. 1 2019
# Added by Guanlin Li

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('switchout')
class SwitchoutSchedule(FairseqLRScheduler):
    """
    The LR schedule introduced in
    [Switchout](https://aclweb.org/anthology/D18-1100).

    Decay every 1000 updates by a fixed factor.
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        self.lr = args.lr[0]  # set to the first 1000 updates
        self.decay_cnt = 0

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        # parser.add_argument('--switchout', '--sw', type=bool, default=False,
        #                     help='force annealing at specified epoch')
        parser.add_argument('--decay-until', type=int, default=4000,
                            help='decay learning rate until certain update number')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        # do nothing after epoch ends
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates >= self.args.decay_until and num_updates % 1000 == 0:
            self.decay_cnt += 1
            self.optimizer.set_lr(self.lr * self.args.lr_shrink ** self.decay_cnt)
        return self.optimizer.get_lr()
