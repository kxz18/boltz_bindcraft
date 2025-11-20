#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import yaml
import time
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
    config_path = os.path.join(config_dir, f'round{rnd}.yaml')
    os.system(f'cp {args.config} {config_path}')
    # update config path
    manifest = data_updater([Path(config_path).expanduser()])
    data_module.manifest = manifest
    model_module.setup_config(config_path)
    # result dir
    result_dir = os.path.join(args.out_dir, 'results')
    os.makedirs(result_dir)

    # Compute structure predictions
    while rnd < model_module.generator_config.max_outer_steps:
        # do structure prediction
        model_module.set_mode_generation(False)
        start = time.time()
        res = trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=True,
        )[0]
        print_log(f'structure prediction elapsed {round(time.time() - start, 2)}s')
        print_log(f'loss: {res["loss_details"]}')
        # save the predictions to another place
        os.system(f'mv {os.path.join(args.out_dir, f"boltz_results_{name}", "predictions", f"round{rnd}")} {result_dir}')

        # do generation
        model_module.set_mode_generation(True)
        res = trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=True,
        )[0]

        print_log(f'after optimization: {res["loss_details"]}')
        print_log(f'outer loop elapsed {round(time.time() - start, 2)}s')
        print()

        rnd += 1
        # update sequence for the next round
        next_config_path = os.path.join(config_dir, f'round{rnd}.yaml')
        new_seqs = update_sequence_to_config(config_path, next_config_path, res['optimized_res_type'])
        config_path = next_config_path
        # update config path
        manifest = data_updater([Path(config_path).expanduser()])
        data_module.manifest = manifest
        model_module.setup_config(config_path)

        # start logging for next round
        print_log('=' * 20 + f'Round {rnd}' + '=' * 20)
        print_log(f'updated sequence: {new_seqs}')


if __name__ == '__main__':
    main(parse())