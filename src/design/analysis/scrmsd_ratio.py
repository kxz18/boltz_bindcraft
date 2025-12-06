#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

from .analysis_utils.load_metrics import load_metrics_for_traj
from .visualize.lineplot import lineplot
from ..utils.logger import print_log


def parse():
    default_out_dir = os.path.join(os.path.dirname(__file__), 'images', os.path.splitext(os.path.basename(__file__))[0])
    parser = argparse.ArgumentParser(description='visualize the scrmsd status for results with multiple trajectores')
    parser.add_argument('--res_dir', type=str, required=True, help='directory of the results')
    parser.add_argument('--tgt_chains', type=str, required=True, help='target chain ids (e.g. AB)')
    parser.add_argument('--lig_chains', type=str, required=True, help='ligand chain ids (e.g. HL)')
    parser.add_argument('--out_dir', type=str, default=default_out_dir, help='output directory')
    return parser.parse_args()


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
    
    # draw figures
    os.makedirs(args.out_dir, exist_ok=True)
    x_name, y_name, ratio_name, hue_name = 'round', 'scRMSD', 'scRMSD<=5.0', 'trajectory'
    data = { x_name: [], y_name: [], ratio_name: [], hue_name: [] }
    data_ratio = { x_name: [], ratio_name: [], hue_name: [] }
    for traj in all_metrics:
        scrmsds = []
        for r in all_metrics[traj]:
            for n in all_metrics[traj][r]:
                met = all_metrics[traj][r][n]
                scrmsds.append(met['af3_scRMSD'])
                data[x_name].append(r)
                data[y_name].append(met['af3_scRMSD'])
                data[hue_name].append(traj)
            data_ratio[x_name].append(r)
            cnt = 0
            for v in scrmsds:
                if v <= 5.0: cnt += 1
            data_ratio[ratio_name].append(cnt / len(scrmsds))
            data_ratio[hue_name].append(traj)
        cnt = 0
        for v in scrmsds:
            if v <= 5.0: cnt += 1
        print_log(f'cnt: {cnt}')
    lineplot(data, x_name, y_name, hue_name, save_path=os.path.join(args.out_dir, 'scrmsd.png'))
    lineplot(data_ratio, x_name, ratio_name, hue_name, save_path=os.path.join(args.out_dir, 'scrmsd_ratio.png'))


if __name__ == '__main__':
    main(parse())