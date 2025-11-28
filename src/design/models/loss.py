#!/usr/bin/python
# -*- coding:utf-8 -*-
from abc import ABC, abstractmethod

import torch

from .info import ComplexInfo
from ..utils import register as R


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
    def __call__(self, dict_out, feats, cplx_info: ComplexInfo):    # loss, record value
        raise NotImplementedError()
    

@R.register('pLDDT')
class pLDDT(Objective):

    def __call__(self, dict_out, feats, cplx_info: ComplexInfo):
        plddt = dict_out['plddt'].mean()
        return 1.0 - plddt, round(plddt.item(), 3)
    

@R.register('ipLDDT')
class ipLDDT(Objective):
    
    def __call__(self, dict_out, feats, cplx_info: ComplexInfo):
        iplddt = dict_out['complex_iplddt'].mean()
        return 1.0 - iplddt, round(iplddt.item(), 3)
    


@R.register('ipTM')
class ipTM(Objective):

    def __call__(self, dict_out, feats, cplx_info: ComplexInfo):
        iptm = dict_out['iptm'].mean()
        return 1.0 - iptm, round(iptm.item(), 3)
    

@R.register('CPipTM')
class CPipTM(Objective):

    def __init__(self, chain_pairs):
        super().__init__()
        self.chain_pairs = chain_pairs

    def __call__(self, dict_out, feats, cplx_info: ComplexInfo):
        iptm = 0
        for a, b in self.chain_pairs:
            if isinstance(a, str): a = cplx_info.chain_orders[a]
            if isinstance(b, str): b = cplx_info.chain_orders[b]
            iptm += dict_out['pair_chains_iptm'][a][b].mean()
            iptm += dict_out['pair_chains_iptm'][b][a].mean()
        iptm = iptm * (0.5 / len(self.chain_pairs))
        return 1.0 - iptm, round(iptm.item(), 3)
    

@R.register('pAE')
class pAE(Objective):
    def __call__(self, dict_out, feats, cplx_info: ComplexInfo):
        pae = dict_out['pae'].mean()
        return pae, round(pae.item(), 3)
    

@R.register('ipDE')
class ipDE(Objective):
    def __call__(self, dict_out, feats, cplx_info: ComplexInfo):
        ipde = dict_out['complex_ipde'].mean()
        return ipde, round(ipde.item(), 3)


@R.register('FWAway')
class FWAway(Objective):

    def __init__(self, binder_chains, contact_th=8.0, k=3):
        super().__init__()
        self.binder_chains = binder_chains
        self.contact_th = contact_th
        self.k = k

    def __call__(self, dict_out, feats, cplx_info: ComplexInfo):
        token_to_center_atom_idx = torch.argmax(feats['token_to_center_atom'], dim=-1)[0]  # [Nres], assume batch size = 1
        center_x = dict_out['sample_atom_coords'][0][token_to_center_atom_idx] # [Nres]
        cplx_info.generate_mask = cplx_info.generate_mask.to(center_x.device)
        binder_mask = torch.tensor([(True if c in self.binder_chains else False) for c in cplx_info.chain_ids], dtype=bool, device=center_x.device)
        target_x, fw_x = center_x[~binder_mask], center_x[binder_mask & (~cplx_info.generate_mask[0])] # [Nres1, 3], #[Nres2, 3]
        dist = torch.norm(target_x[:, None] - fw_x[None, :], dim=-1)    # [Nres1, Nres2]
        mink_dist = torch.topk(dist, k=self.k, dim=-1, largest=False)[0]    # [Nres1, k]
        contact = mink_dist < self.contact_th
        cnt = contact.sum(-1).bool().sum().item()
        if cnt == 0:
            return 0, (self.contact_th, cnt)
        # avg_contact_dist = mink_dist[contact].mean()
        # loss = self.contact_th - avg_contact_dist
        # return loss, (round(avg_contact_dist.item(), 2), cnt)
        contact_dist = mink_dist[contact].sum()
        loss = self.contact_th * contact.sum() - contact_dist
        return loss, (round(contact_dist.item(), 2), cnt)
    

@R.register('Epitope')
class Epitope(Objective):

    def __init__(self, epitope: list, k: int=3):
        super().__init__()
        self.epitope = epitope  # e.g. [[A, 105], [A, 117], [A, 168], [B, 200]]
        self.k = k
    
    def __call__(self, dict_out, feats, cplx_info: ComplexInfo):
        # prepare
        epi_dict = {}
        for c, pos in self.epitope:
            pos = int(pos)
            if c not in epi_dict: epi_dict[c] = {}
            epi_dict[c][pos] = True
        token_to_center_atom_idx = torch.argmax(feats['token_to_center_atom'], dim=-1)[0]  # [Nres], assume batch size = 1
        center_x = dict_out['sample_atom_coords'][0][token_to_center_atom_idx] # [Nres]
        cplx_info.generate_mask = cplx_info.generate_mask.to(center_x.device)
        epi_mask = []
        pos, last_c = 0, None
        for c in cplx_info.chain_ids:
            if c != last_c: pos = 0
            last_c = c
            if (c in epi_dict) and (pos in epi_dict[c]): epi_mask.append(True)
            else: epi_mask.append(False)
            pos += 1
        epi_mask = torch.tensor(epi_mask, dtype=bool, device=center_x.device)
        epi_x, gen_x = center_x[epi_mask], center_x[cplx_info.generate_mask[0]] # [Nres1, 3], #[Nres2, 3]
        dist = torch.norm(epi_x[:, None] - gen_x[None, :], dim=-1)    # [Nres1, Nres2]
        mink_dist = torch.topk(dist, k=self.k, dim=-1, largest=False)[0]
        avg_dist = mink_dist.mean()
        return avg_dist, round(avg_dist.item(), 2)