#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json

from ...af3.af3_utils import load_confidences, get_scRMSD


def get_best(name, vals, largest=None):
    if largest is None:
        largest = {
            'pLDDT': True,
            'CPipTM': True,
            'FWAway': (True, False),
            'FWAwayExp': (True, False),
            'Epitope': False,
            'total': False,
            'ChainpLDDT': True,
            'GenPartpLDDT': True
        }[name]
    if isinstance(largest, tuple):
        return [get_best(name, [t[i] for t in vals], largest=largest[i]) for i in range(len(largest))]
    else:
        return max(vals) if largest else min(vals)


def load_history(path):
    with open(path, 'r') as fin: data = json.load(fin)
    history_seqs, history_metrics = {}, {}
    for name in data:
        r, n = name.lstrip('round').split('_')
        r, n = int(r), int(n)   # round, which candidate
        if r not in history_seqs: history_seqs[r], history_metrics[r] = {}, {}
        history_seqs[r][n], history_metrics[r][n] = data[name]
    for r in history_metrics:
        best_metrics = {}
        for metric_name in history_metrics[r][0]:
            best_metrics[metric_name] = get_best(metric_name, [history_metrics[r][i][metric_name] for i in history_metrics[r]])
        history_metrics[r]['best'] = best_metrics
    return history_seqs, history_metrics


def load_af3_metrics(candidate_dir, tgt_chains, lig_chains, iptm_row_cols):
    save_path = os.path.join(candidate_dir, 'af3', 'metrics.json')
    try:
        af3_confidences = json.load(open(save_path, 'r'))
        return af3_confidences
    except Exception: pass
    # process
    confidence_path = os.path.join(candidate_dir, 'af3', 'output', 'AF3', 'AF3_summary_confidences.json')
    af3_confidences = load_confidences(confidence_path, tgt_chains, lig_chains, iptm_row_cols)
    if af3_confidences is None: return None
    af3_model_path = os.path.join(candidate_dir, 'af3', 'output', 'AF3', 'AF3_model.cif')
    for fname in os.listdir(candidate_dir):
        if fname.endswith('.cif'): break
    ref_path = os.path.join(candidate_dir, fname)
    sc_rmsd, _ = get_scRMSD(ref_path, af3_model_path, tgt_chains, lig_chains)
    af3_confidences['scRMSD'] = sc_rmsd
    json.dump(af3_confidences, open(save_path, 'w'), indent=2)
    return af3_confidences


def load_metrics_for_traj(traj_dir, tgt_chains, lig_chains):
    _, history_metrics = load_history(os.path.join(traj_dir, 'results', 'history.json'))
    chain2af3idx = {}
    for i, c in enumerate(sorted(tgt_chains + lig_chains)):
        chain2af3idx[c] = i
    iptm_row_cols = []
    for c1 in tgt_chains:
        for c2 in lig_chains: iptm_row_cols.append((chain2af3idx[c1], chain2af3idx[c2]))
    res_dir = os.path.join(traj_dir, 'results')
    af3_metrics = {}
    for rnd_dir in os.listdir(res_dir):
        if not rnd_dir.startswith('round'): continue
        rnd = int(rnd_dir.lstrip('round'))
        af3_metrics[rnd] = {}
        for n_dir in os.listdir(os.path.join(res_dir, rnd_dir)):
            if not n_dir.isdigit(): continue
            n = int(n_dir)
            n_dir = os.path.join(res_dir, rnd_dir, n_dir)
            if not os.path.isdir(n_dir): continue
            loaded_data = load_af3_metrics(n_dir, tgt_chains, lig_chains, iptm_row_cols)
            if loaded_data is None: continue
            af3_metrics[rnd][n] = loaded_data
    # merge metrics
    all_metrics = {}
    for r in af3_metrics:
        all_metrics[r] = {}
        for n in af3_metrics[r]:
            met = {}
            met.update(history_metrics[r][n])
            for name in af3_metrics[r][n]:
                met['af3_' + name] = af3_metrics[r][n][name]
            all_metrics[r][n] = met
            
    return all_metrics