#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import random


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_rng_state():
    return {
         'torch_cpu': torch.get_rng_state(),
         'numpy': np.random.get_state(),
         'torch_cuda': torch.cuda.get_rng_state_all(),
         'random': random.getstate()
    }


def set_rng_state(state_dict):
     torch.set_rng_state(state_dict['torch_cpu'])
     torch.cuda.set_rng_state_all(state_dict['torch_cuda'])
     np.random.set_state(state_dict['numpy'])
     random.setstate(state_dict['random'])
