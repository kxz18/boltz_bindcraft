#!/usr/bin/python
# -*- coding:utf-8 -*-
#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import numpy as np

from ..data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from ..evaluation.rmsd import kabsch, compute_rmsd
from ..utils.logger import print_log


def load_confidences(json_path, tgt_chains, lig_chains, iptm_row_cols=None):
    if not os.path.exists(json_path):
        print_log(f'{json_path} not exists when loading af3 confidences', level='WARN')
        return None
    tgt_chains, lig_chains = set(tgt_chains), set(lig_chains)

    # from confidence summary
    item_summary = json.load(open(json_path, 'r'))

    # from full details
    full_detail_json_path = json_path.replace('summary_', '')
    item = json.load(open(full_detail_json_path, 'r'))
    # plddt
    atom_chain_ids = item['atom_chain_ids']
    atom_plddts = item['atom_plddts']
    binder_plddts = [plddt for plddt, c in zip(atom_plddts, atom_chain_ids) if c in lig_chains]

    # ipAE
    token_chain_ids, token_pae = item['token_chain_ids'], item['pae']
    ipae = []
    for i, row in enumerate(token_pae):
        for j, val in enumerate(row):
            ci, cj = token_chain_ids[i], token_chain_ids[j]
            if (ci in tgt_chains and cj in lig_chains) or (ci in lig_chains and cj in tgt_chains):
                ipae.append(val)

    # iptm
    if iptm_row_cols is not None:
        chain_pair_iptm = item_summary['chain_pair_iptm']
        iptm = [chain_pair_iptm[row][col] for row, col in iptm_row_cols]
        iptm = sum(iptm) / len(iptm)
    else: iptm = item_summary['iptm']   # overall iptm

    return {
        'iptm': item_summary['iptm'],
        'ptm': item_summary['ptm'],
        'ranking_score': item_summary['ranking_score'],
        'plddt': sum(atom_plddts) / len(atom_plddts) if len(atom_plddts) > 0 else None,
        'binder_plddt': sum(binder_plddts) / len(binder_plddts) if len(binder_plddts) > 0 else None,
        'ipae': sum(ipae) / len(ipae) if len(ipae) > 0 else None
    }


def get_scRMSD(ref_path, model_path, tgt_chains, lig_chains, gen_mask=None, align_by_target=True):
    ref_cplx = mmcif_to_complex(ref_path, selected_chains=tgt_chains + lig_chains)
    model_cplx = mmcif_to_complex(model_path, selected_chains=tgt_chains + lig_chains)

    # get CA coordinates
    def get_ca_coords(mol):
        coords = []
        for block in mol:
            for atom in block:
                if atom.name == 'CA': coords.append(atom.get_coord())
        return coords

    # get CA coordinates of the target
    ref_tgt_ca, model_tgt_ca = [], []
    for c in tgt_chains: ref_tgt_ca.extend(get_ca_coords(ref_cplx[c]))
    for c in tgt_chains: model_tgt_ca.extend(get_ca_coords(model_cplx[c]))
    ref_tgt_ca, model_tgt_ca = np.array(ref_tgt_ca), np.array(model_tgt_ca)

    # get CA coordinates of the ligand
    ref_lig_ca, model_lig_ca = [], []
    for c in lig_chains:
        ref_lig_ca.extend(get_ca_coords(ref_cplx[c]))
        model_lig_ca.extend(get_ca_coords(model_cplx[c]))
    ref_lig_ca, model_lig_ca = np.array(ref_lig_ca), np.array(model_lig_ca)
    
    if align_by_target:
        # get transformation matrix
        _, rotation, t = kabsch(model_tgt_ca, ref_tgt_ca)
        # transform
        model_lig_ca_aligned = np.dot(model_lig_ca, rotation) + t
    else: model_lig_ca_aligned, _, _ = kabsch(model_lig_ca, ref_lig_ca)

    sc_rmsd = compute_rmsd(ref_lig_ca, model_lig_ca_aligned)
    if gen_mask is not None:
        gen_mask = np.array(gen_mask, dtype=bool)
        assert len(gen_mask) == len(model_lig_ca_aligned)
        gen_sc_rmsd = compute_rmsd(ref_lig_ca[gen_mask], model_lig_ca_aligned[gen_mask])
    else: gen_sc_rmsd = None
    
    return sc_rmsd, gen_sc_rmsd
