#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
from functools import partial

from scipy.stats import pearsonr
from matplotlib import pyplot as plt

from .analysis_utils.load_metrics import load_metrics_for_traj
from .analysis_utils.helpers import get_metric_val, flatten_metric_names, get_topk_val
from .visualize.lineplot import lineplot
from .visualize.scatterplot import scatterplot
from .visualize.heatmap import heatmap
from ..utils.logger import print_log


def parse():
    default_out_dir = os.path.join(os.path.dirname(__file__), 'images', os.path.splitext(os.path.basename(__file__))[0])
    parser = argparse.ArgumentParser(description='visualize the scrmsd status for results with multiple trajectores')
    parser.add_argument('--res_dir', type=str, required=True, help='directory of the results')
    parser.add_argument('--tgt_chains', type=str, required=True, help='target chain ids (e.g. AB)')
    parser.add_argument('--lig_chains', type=str, required=True, help='ligand chain ids (e.g. HL)')
    parser.add_argument('--out_dir', type=str, default=default_out_dir, help='output directory')
    return parser.parse_args()


def avg(vals, name):
    return [sum(vals) / len(vals)]


def round_avg(traj_data, metric_name):
    vals, rounds = [], []
    for r in traj_data:
        for n in traj_data[r]:
            vals.append(get_metric_val(traj_data[r][n], metric_name))
            rounds.append(r)
    return vals, rounds


def round_best(traj_data, metric_name):
    vals, rounds = [], []
    for r in traj_data:
        r_vals = []
        for n in traj_data[r]: r_vals.append(get_metric_val(traj_data[r][n], metric_name))
        if len(r_vals) == 0: continue
        vals.append(get_topk_val(r_vals, metric_name, topk=1)[0])
        rounds.append(r)
    return vals, rounds


def accumulative(traj_data, metric_name, agg_func=avg):
    r2data = {}
    for r in traj_data:
        r2data[r] = []
        for n in traj_data[r]:
            r2data[r].append(get_metric_val(traj_data[r][n], metric_name))
    history, vals, rounds = [], [], []
    for r in sorted(list(r2data.keys())):
        history.extend(r2data[r])
        if len(history) == 0: continue
        agg_vals = agg_func(history, metric_name)
        vals.extend(agg_vals)
        rounds.extend([r for _ in agg_vals])
    return vals, rounds



def main(args):
    all_metrics = {}
    for traj in os.listdir(args.res_dir):
        if not traj.startswith('trajectory'): continue
        traj_dir = os.path.join(args.res_dir, traj)
        print_log(f'Loading results from {traj_dir}')
        all_metrics[traj] = load_metrics_for_traj(traj_dir, list(args.tgt_chains), list(args.lig_chains))

    for traj in all_metrics:
        for r in all_metrics[traj]:
            for n in all_metrics[traj][r]:
                print_log(f'example data structure from {traj} {r} {n}: {all_metrics[traj][r][n]}')
                break
            break
        break
    
    # prepare 
    th, metric_names = 5.0, None
    cnt = 0
    for traj in all_metrics:
        for r in all_metrics[traj]:
            for n in all_metrics[traj][r]:
                all_metrics[traj][r][n][f'scRMSD<={th}'] = int(all_metrics[traj][r][n]['af3_scRMSD'] <= th)
                if metric_names is None: metric_names = flatten_metric_names(all_metrics[traj][r][n])
                cnt += 1
    os.makedirs(args.out_dir, exist_ok=True)
    print_log(f'Total number of candidates: {cnt}')

    # draw different metrics
    # 1. round-based avg
    # 2. accumulative avg
    # 3. round-based best
    # 4. accumulative best
    # 5. accumulative topk
    topk = 5
    mode2func = {
        'round_avg': round_avg,
        'round_best': round_best,
        'accum_avg': accumulative,
        'accum_best': partial(accumulative, agg_func=get_topk_val),
        f'accum_top{topk}': partial(accumulative, agg_func=partial(get_topk_val, topk=topk))
    }

    x_name, hue_name = 'round', 'trajectory'
    for mode in mode2func:
        out_dir = os.path.join(args.out_dir, mode)
        os.makedirs(out_dir, exist_ok=True)
        print_log(f'Drawing figures in mode {mode} under {out_dir}')
        data_func = mode2func[mode]
        for y_name in metric_names:
            data = { x_name: [], y_name: [], hue_name: [] }
            for traj in all_metrics:
                y, x = data_func(all_metrics[traj], y_name)
                try: y, x = data_func(all_metrics[traj], y_name)
                except Exception:
                    print_log(f'Failed to extract data for {traj}/{y_name} in mode {mode}', level='WARN')
                    continue
                data[x_name].extend(x)
                data[y_name].extend(y)
                data[hue_name].extend([traj for _ in x])
            lineplot(data, x_name, y_name, hue_name, save_path=os.path.join(out_dir, f'{y_name}.png'))

    # analyze correlation between different metrics
    data = []
    for met1 in metric_names:
        data.append([])
        vals1 = []
        for traj in all_metrics:
            for r in all_metrics[traj]:
                for n in all_metrics[traj][r]:
                    vals1.append(get_metric_val(all_metrics[traj][r][n], met1))
        for met2 in metric_names:
            vals2 = []
            for traj in all_metrics:
                for r in all_metrics[traj]:
                    for n in all_metrics[traj][r]:
                        vals2.append(get_metric_val(all_metrics[traj][r][n], met2))
            data[-1].append(pearsonr(vals1, vals2).correlation)
    def post_edit(ax):
        plt.tight_layout()
    plt.figure(figsize=(16, 12))
    heatmap(
        data, save_path=os.path.join(args.out_dir, 'corr.png'), annot=True, fmt='.2f', post_edit_func=post_edit,
        xticklabels=metric_names, yticklabels=metric_names
    )

    # scrmsd - boltz plddt - af3 plddt (color) (or iptm)

if __name__ == '__main__':
    main(parse())