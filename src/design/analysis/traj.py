#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse

from .visualize.lineplot import lineplot

from ..utils.logger import print_log


def parse():
    default_out_dir = os.path.join(os.path.dirname(__file__), 'images', os.path.splitext(os.path.basename(__file__))[0])
    parser = argparse.ArgumentParser(description='visualize the loss trajectory')
    parser.add_argument('--res_dir', type=str, required=True, help='directory of the results')
    parser.add_argument('--out_dir', type=str, default=default_out_dir, help='output directory')
    return parser.parse_args()


def get_best(name, vals, largest=None):
    if largest is None:
        largest = {
            'pLDDT': True,
            'CPipTM': True,
            'FWAway': (True, False),
            'Epitope': False,
            'total': False
        }[name]
    if isinstance(largest, tuple):
        return [get_best(name, [t[i] for t in vals], largest=largest[i]) for i in range(len(largest))]
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


def count_seq_freq(history_seqs):
    seqs = {}
    for r in history_seqs:
        if r == 0: continue     # 0 is initialization
        for n in history_seqs[r]:
            for chain_name in history_seqs[r][n]:
                if chain_name not in seqs: seqs[chain_name] = []
                seqs[chain_name].append(history_seqs[r][n][chain_name])
    
    freq_dict = {}
    for chain_name in seqs:
        freq_dict[chain_name] = {}
        for seq in seqs[chain_name]:
            if seq not in freq_dict[chain_name]: freq_dict[chain_name][seq] = 0
            freq_dict[chain_name][seq] += 1
    return freq_dict


def load_traj(dir):
    all_trajs = {}
    for rname in os.listdir(dir):
        traj_path = os.path.join(dir, rname, 'loss_traj.json')
        if not os.path.exists(traj_path): continue
        with open(traj_path, 'r') as fin: loss_traj = json.load(fin)
        r = int(rname.lstrip('round'))
        all_trajs[r] = loss_traj
    return all_trajs


def get_metric_val(d, name):
    if name == 'FWAway_dist': return d['FWAway'][0]
    elif name == 'FWAway_cnt': return d['FWAway'][1]
    return d[name]


def main(args):
    # load history
    history_seqs, history_metrics = load_history(os.path.join(args.res_dir, 'results', 'history.json'))

    # count sequence uniqueness
    seq_freq = count_seq_freq(history_seqs)
    for chain_name in seq_freq:
        freq = seq_freq[chain_name]
        uniq = len(freq) / sum(list(freq.values())) * 100
        print_log(f'Chain {chain_name}: Uniqueness {round(uniq, 2)}%')
        topk = 5
        print_log(f'\ttop {topk} sequence:')
        for key in sorted(freq, key=lambda seq: freq[seq], reverse=True)[:topk]:
            print_log(f'\t{key} frequency {freq[key]}')
    
    # load trajectories in each round
    traj = load_traj(os.path.join(args.res_dir, 'results'))

    # create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # get figure data
    metrics = { name: [name] for name in traj[1][0] }
    if 'FWAway' in metrics:
        metrics['FWAway'] = ('FWAway_dist', 'FWAway_cnt')
    for metric_type in metrics:
        for metric_name in metrics[metric_type]:
            x_name = 'round'
            data = {
                metric_name: [],
                x_name: []
            }
            for r in history_metrics:
                if r not in traj: continue
                # data[x_name].append(r)
                # data[metric_name].append(get_metric_val(history_metrics[r]['best'], metric_name))
                for i, val in enumerate(traj[r]):
                    data[x_name].append(r - 1 + i / len(traj[r]))
                    data[metric_name].append(get_metric_val(val, metric_name))
            # draw picture
            lineplot(data, x_name, metric_name, save_path=os.path.join(args.out_dir, metric_name + '.png'))
            print_log(f'Finished curves for {metric_name}')

if __name__ == '__main__':
    main(parse())