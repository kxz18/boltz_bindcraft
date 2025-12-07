#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import yaml
import time
import json
import random
import argparse
from pathlib import Path

import ray
import numpy as np

from .utils.logger import print_log
from .utils.boltz_utils import prepare_boltz2
from .af3.rectification import af3_rectification



def parse():
    parser = argparse.ArgumentParser(description='Design with diffusion gradients')
    parser.add_argument('--config', type=str, required=True, help='Path to the configurations')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--ckpt_dir', type=str, default='~/.boltz', help='Directory of the boltz checkpoints')
    parser.add_argument('--max_num_trajectories', type=int, default=None, help='Maximum number of trajectories. If set to None, the program will keep running until a keyboard interrupt')
    parser.add_argument('--af3_msa_config', type=str, default=None, help='Path to MSA configurations for AF3 rectification')
    return parser.parse_args()



def update_sequence_to_config(src_path, dst_path, res_type):
    with open(src_path, 'r') as fin: config = yaml.safe_load(fin)
    offset = 0
    masks = config['generator']['masks']
    new_seqs = {}
    for item in config['sequences']:
        assert 'protein' in item
        chain_id = item['protein']['id']
        new_seq = ''.join(res_type[offset:offset+len(item['protein']['sequence'])])
        offset += len(item['protein']['sequence'])
        if chain_id not in masks:
            continue
        item['protein']['sequence'] = new_seq
        new_seqs[chain_id] = new_seq
    
    with open(dst_path, 'w') as fout: yaml.dump(config, fout)

    return new_seqs


def loss_to_prob(losses):
    losses = np.array(losses)
    p = losses - losses.mean()
    p = 1 / np.exp(p)
    p = p / p.sum()
    return p


def run_design(config, out_dir, ckpt_dir, af3_msa_config):
    start = time.time()
    trainer, model_module, data_module, data_updater = prepare_boltz2(
        data = config,
        out_dir = out_dir,
        cache = ckpt_dir,
        silent = True
    )
    print_log(f'Setting up additional configurations')
    model_module.init()
    model_module.eval()
    model_module.setup_config(config)
    model_module.enable_param_gradients()
    model_module.set_mode_generation(True)
    print_log(f'Preparation elapsed {time.time() - start}s')
    if trainer is None: return

    if model_module.generator_config.af3_rect_freq > 0:
        ray.init(
            include_dashboard=False,
            logging_level='error',
            ignore_reinit_error=True,
        )

    name = os.path.splitext(os.path.basename(config))[0]

    rnd = 0
    config_dir = os.path.join(out_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f'round{rnd}_0.yaml')
    os.system(f'cp {config} {config_path}')
    # update config path
    manifest = data_updater([Path(config_path).expanduser()])
    data_module.manifest = manifest
    model_module.setup_config(config_path)
    # result dir
    result_dir = os.path.join(out_dir, 'results')
    os.makedirs(result_dir)

    loss_traj, new_res_types, history, history_configs = None, [None], {}, {}
    patience = model_module.generator_config.converge_patience
    # Compute structure predictions
    while rnd < model_module.generator_config.max_outer_steps:
        # do structure prediction
        rnd_res_dir = os.path.join(result_dir, f'round{rnd}')
        os.makedirs(rnd_res_dir, exist_ok=True)
        i2config, i2loss = {}, {}
        for i, res_type in enumerate(new_res_types):
            if res_type is not None:    # update sequence in the configuration
                next_config_path = os.path.join(config_dir, f'round{rnd}_{i}.yaml')
                new_seqs = update_sequence_to_config(config_path, next_config_path, res_type)
                # config_path = next_config_path
                # update config path
                manifest = data_updater([Path(next_config_path).expanduser()])
                data_module.manifest = manifest
                model_module.setup_config(next_config_path)
                print_log(f'evaluating {i}-th seqs: {new_seqs}')
            else: next_config_path, new_seqs = config_path, 'init'
            i2config[i] = next_config_path
            # structure prediction
            model_module.set_mode_generation(False)
            start = time.time()
            res = trainer.predict(
                model_module,
                datamodule=data_module,
                return_predictions=True,
            )[0]
            print_log(f'structure prediction elapsed {round(time.time() - start, 2)}s')
            print_log(f'loss: {res["loss_details"]}')
            i2loss[i] = res['loss_details']
            history[f'round{rnd}_{i}'] = (new_seqs, res['loss_details'])
            history_configs[f'round{rnd}_{i}'] = next_config_path
            # save the predictions to another place
            os.system(f'mv {os.path.join(out_dir, f"boltz_results_{name}", "predictions", f"round{rnd}_{i}")} {os.path.join(rnd_res_dir, str(i))}')
        # save the loss trajectory
        if loss_traj is not None:
            with open(os.path.join(rnd_res_dir, 'loss_traj.json'), 'w') as fout: json.dump(loss_traj, fout, indent=2)
        # save history records
        history = dict(sorted(history.items(), key=lambda x: x[1][1]['total'])) # sort by loss
        with open(os.path.join(result_dir, 'history.json'), 'w') as fout: json.dump(history, fout, indent=2)
        topk_history = list(history.keys())[:model_module.generator_config.history_best_topk]

        # check convergence
        updated = (rnd == 0)    # skip check for the first round
        for history_name in topk_history:
            if f'round{rnd}_' in history_name: updated = True
        if updated: patience = model_module.generator_config.converge_patience
        else: patience -= 1
        print_log(f'Patience: {patience}')
        if patience <= 0:
            print_log(f'Algorithm converged. Early stop.')
            break

        # use the best one for next round
        config_path = os.path.join(config_dir, f'round{rnd}.yaml')
        af3_rect_freq = model_module.generator_config.af3_rect_freq
        if (af3_rect_freq > 0) and (rnd > 0) and (rnd % af3_rect_freq == 0):
            # AF3 rectification
            print_log(f'Round {rnd}, entering AF3 orthogonal rectification')
            rect_topk = model_module.generator_config.sample_k
            sel_name, scrmsd = af3_rectification(history, history_configs, rect_topk, out_dir, af3_msa_config)
            os.system(f'cp {history_configs[sel_name]} {config_path}')
            print_log(f'Using {sel_name}, the one with best scRMSD ({round(scrmsd, 2)}) among top-{rect_topk} for the next round')
        elif model_module.generator_config.use_history_best:
            # use history best for next round
            probs = loss_to_prob([history[sel_name][1]['total'] for sel_name in topk_history])
            print_log(f'Top-{len(topk_history)} probabilites as the starter for the next round: {[round(p, 2) for p in probs.tolist()]}')
            sel = np.random.choice(np.arange(len(topk_history)), p=probs, size=1)[0]
            sel_name = topk_history[sel]
            os.system(f'cp {history_configs[sel_name]} {config_path}')
            print_log(f'Using the {sel}-th best one in history ({sel_name}) with loss: {history[sel_name][1]}')
        else:
            best_i = min(i2loss, key=lambda i: i2loss[i]['total'])
            os.system(f'cp {i2config[best_i]} {config_path}')
            print_log(f'Best one in this loop ({best_i}) with loss: {i2loss[i]}')
        # get topk in the history
        topk = model_module.generator_config.print_history_topk
        print_log(f'History top {topk}')
        for i, cid in enumerate(history):
            if i == topk: break
            print_log(f'\t{i}. {cid} {history[cid]}')
        # update config path
        manifest = data_updater([Path(config_path).expanduser()])
        data_module.manifest = manifest
        model_module.setup_config(config_path)
        print_log(f'Using configuration from {config_path} for optimization')

        # do generation
        model_module.set_mode_generation(True)
        res = trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=True,
        )[0]

        loss_traj = res['loss_traj']    # update trajectory
        print_log(f'outer loop elapsed {round(time.time() - start, 2)}s')
        print()

        rnd += 1
        model_module.increase_outer_loop()
        # update sequence for the next round
        new_res_types = res['optimized_res_type']

        # start logging for next round
        print_log('=' * 20 + f'Round {rnd}' + '=' * 20)


def main(args):
    traj_cnt = 0
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        base_config = os.path.join(args.out_dir, 'base_config.yaml')
        os.system(f'cp {args.config} {base_config}')
        while (args.max_num_trajectories is None) or (traj_cnt < args.max_num_trajectories):
            print_log('=' * 30 + f' Running trajectory {traj_cnt} ' + '=' * 30)
            out_dir = os.path.join(args.out_dir, f'trajectory{traj_cnt}')
            run_design(base_config, out_dir, args.ckpt_dir, args.af3_msa_config)
            # get some spaces between trajectories
            for _ in range(10): print()
            traj_cnt += 1
    except KeyboardInterrupt:
        print_log(f'Stopping due to keyboard interrupt')

if __name__ == '__main__':
    main(parse())