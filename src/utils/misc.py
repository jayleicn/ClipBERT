"""
modified from UNITER
"""
import json
import random
import sys

import torch
import numpy as np


class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zero_none_grad(model):
    for p in model.parameters():
        if p.grad is None and p.requires_grad:
            p.grad = p.data.new(p.size()).zero_()
