#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import yaml
import time
import json
import shutil
import tempfile
import argparse
from pathlib import Path

from .utils.logger import print_log
from .utils.boltz_utils import prepare_boltz2



def parse():
    parser = argparse.ArgumentParser(description='Design with diffusion gradients')
    parser.add_argument('--config', type=str, required=True, help='Path to the configurations')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--ckpt_dir', type=str, default='~/.boltz', help='Directory of the boltz checkpoints')
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


def main(args):
    start = time.time()
    trainer, model_module, data_module, data_updater = prepare_boltz2(
        data = args.config,
        out_dir = args.out_dir,
        cache = args.ckpt_dir,
        silent = True
    )
    print_log(f'Setting up additional configurations')
    model_module.eval()
    model_module.setup_config(args.config)
    model_module.enable_param_gradients()
    model_module.set_mode_generation(True)
    print_log(f'Preparation elapsed {time.time() - start}s')
    if trainer is None: return

    name = os.path.splitext(os.path.basename(args.config))[0]

    rnd = 0
    config_dir = os.path.join(args.out_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f'round{rnd}_0.yaml')
    os.system(f'cp {args.config} {config_path}')
    # update config path
    manifest = data_updater([Path(config_path).expanduser()])
    data_module.manifest = manifest
    model_module.setup_config(config_path)
    # result dir
    result_dir = os.path.join(args.out_dir, 'results')
    os.makedirs(result_dir)

    loss_traj, new_res_types, history, history_configs = None, [None], {}, {}
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
            os.system(f'mv {os.path.join(args.out_dir, f"boltz_results_{name}", "predictions", f"round{rnd}_{i}")} {os.path.join(rnd_res_dir, str(i))}')
        # save the loss trajectory
        if loss_traj is not None:
            with open(os.path.join(rnd_res_dir, 'loss_traj.json'), 'w') as fout: json.dump(loss_traj, fout, indent=2)
        # save history records
        history = dict(sorted(history.items(), key=lambda x: x[1][1]['total'])) # sort by loss
        with open(os.path.join(result_dir, 'history.json'), 'w') as fout: json.dump(history, fout, indent=2)

        # use the best one for next round
        config_path = os.path.join(config_dir, f'round{rnd}.yaml')
        if model_module.generator_config.use_history_best:
            # use history best for next round
            best_i = min(history, key=lambda i: history[i][1]['total'])
            os.system(f'cp {history_configs[best_i]} {config_path}')
            print_log(f'Best one in history ({best_i}) with loss: {history[best_i][1]}')
        else:
            best_i = min(i2loss, key=lambda i: i2loss[i]['total'])
            os.system(f'cp {i2config[best_i]} {config_path}')
            print_log(f'Best one in this loop ({best_i}) with loss: {i2loss[i]}')
        # get topk in the history
        topk = 5
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
        # print_log(f'after optimization: {res["loss_details"]}')
        print_log(f'outer loop elapsed {round(time.time() - start, 2)}s')
        print()

        rnd += 1
        model_module.increase_outer_loop()
        # update sequence for the next round
        new_res_types = res['optimized_res_type']
        # next_config_path = os.path.join(config_dir, f'round{rnd}.yaml')
        # new_seqs = update_sequence_to_config(config_path, next_config_path, res['optimized_res_type'])
        # config_path = next_config_path
        # # update config path
        # manifest = data_updater([Path(config_path).expanduser()])
        # data_module.manifest = manifest
        # model_module.setup_config(config_path)

        # start logging for next round
        print_log('=' * 20 + f'Round {rnd}' + '=' * 20)
        # print_log(f'updated sequence: {new_seqs}')


if __name__ == '__main__':
    main(parse())