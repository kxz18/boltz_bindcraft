#!/usr/bin/python
# -*- coding:utf-8 -*-
from ..utils import register as R

from abc import ABC, abstractmethod


def parse_losses(config):
    losses = {}
    for c in config:
        w = c.pop('weight')
        loss = R.construct(c)
        losses[loss.name] = (w, loss)
    return losses


class Objective(ABC):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, dict_out, generate_mask):    # loss, record value
        raise NotImplementedError()
    

@R.register('pLDDT')
class pLDDT(Objective):

    def __call__(self, dict_out, generate_mask):
        plddt = dict_out['plddt'].mean()
        return 1.0 - plddt, plddt.item()
    

@R.register('ipTM')
class ipTM(Objective):

    def __call__(self, dict_out, generate_mask):
        iptm = dict_out['iptm'].mean()
        return 1.0 - iptm, iptm.item()
    
@R.register('CPipTM')
class CPipTM(Objective):

    def __init__(self, chain_pairs):
        super().__init__()
        self.chain_pairs = chain_pairs

    def __call__(self, dict_out, generate_mask):
        iptm = 0
        for a, b in self.chain_pairs:
            iptm += dict_out['pair_chains_iptm'][a][b].mean()
            iptm += dict_out['pair_chains_iptm'][b][a].mean()
        iptm = iptm * (0.5 / len(self.chain_pairs))
        return 1.0 - iptm, iptm.item()