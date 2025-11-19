#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import yaml
import time
import tempfile
import argparse
from pathlib import Path

import torch

from .utils.logger import print_log
from .utils.boltz_utils import prepare_boltz2



def parse():
    parser = argparse.ArgumentParser(description='Design with diffusion gradients')
    parser.add_argument('--config', type=str, required=True, help='Path to the configurations')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--ckpt_dir', type=str, default='~/.boltz', help='Directory of the boltz checkpoints')
    return parser.parse_args()




def main(args):
    start = time.time()
    trainer, model_module, data_module, data_updater = prepare_boltz2(
        data = args.config,
        out_dir = args.out_dir,
        cache = args.ckpt_dir,
    )
    print_log(f'Setting up additional configurations')
    add_configs = {}
    model_module.eval()
    model_module.setup_config(**add_configs)
    model_module.enable_param_gradients()
    print_log(f'Preparation elapsed {time.time() - start}s')
    if trainer is None: return

    # Compute structure predictions
    start = time.time()
    res = trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=True,
    )[0]
    print(res.keys())
    print(res['gradient'])
    # res['iptm'].backward()
    print_log(f'Diffusion elapsed {time.time() - start}s')

    # save the predictions to another place
    os.system(f'mv {os.path.join(args.out_dir, "boltz_results_input", "predictions")} {os.path.join(args.out_dir, "boltz_results_input", "predictions_old")}')

    # try to change the sequence
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as fout:
        c = yaml.safe_load(open(args.config))
        seq = c['sequences'][1]['protein']['sequence']
        c['sequences'][1]['protein']['sequence'] = seq[0] + 'W' + seq[2:]
        yaml.dump(c, fout)
        start = time.time()
        manifest = data_updater([Path(fout.name).expanduser()])
        data_module.manifest = manifest
        print_log(f'data update elapsed {time.time() - start}s')
    
    res = trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=True,
    )[0]

if __name__ == '__main__':
    main(parse())