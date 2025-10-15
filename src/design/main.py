#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
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
    trainer, model_module, data_module = prepare_boltz2(
        data = args.config,
        out_dir = args.out_dir,
        cache = args.ckpt_dir,
    )
    print_log(f'Preparation elapsed {time.time() - start}s')

    # Compute structure predictions
    start = time.time()
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )
    print_log(f'Diffusion elapsed {time.time() - start}s')


if __name__ == '__main__':
    main(parse())