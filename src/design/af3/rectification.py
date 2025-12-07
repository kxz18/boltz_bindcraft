#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import gc
import time
import json

import ray
import yaml
import shutil
import torch

from .constructor import construct_input_json
from .af3_utils import get_scRMSD
from .struct_server import form_task
from ..utils.logger import print_log


def _run_af3(config_path, template_dir, template_history, chain2msa_paths, out_dir):
    # get sequences
    config = yaml.safe_load(open(config_path, 'r'))
    # form json input for AF3
    chain2template_cif = {}
    for item in config['templates']:
        for c1, c2 in zip(item['chain_id'], item['template_id']):
            chain2template_cif[c1] = (c2, item['cif'])
    construct_input_json(
        chain2seqs={ item['protein']['id']: item['protein']['sequence'] for item in config['sequences'] },
        chain2msa_paths=chain2msa_paths,
        chain2template_cif=chain2template_cif,
        template_dir=template_dir,
        template_history=template_history,
        task_name='AF3',
        out_dir=out_dir,
        n_seeds=5   # stable results
    )
    # run task with ray
    task = form_task(os.path.join(out_dir, 'AF3.json'), 'AF3')
    futures = [task]
    while len(futures) > 0:
        done_ids, futures = ray.wait(futures, num_returns=1, timeout=1)
        if len(done_ids) == 0: time.sleep(10)   # not any tasks finished yet
        for done_id in done_ids:
            task = ray.get(done_id)
            print_log(f'{task.id} finished. Elapsed {round(task.elapsed_time, 2)}s. Exit code: {task.exit_code}.')


def af3_rectification(sorted_history, history_configs, topk, root_dir, msa_config=None):
    # first cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # template config
    template_dir = os.path.join(root_dir, 'af3_templates')
    os.makedirs(template_dir, exist_ok=True)
    template_history_path = os.path.join(template_dir, 'template_history.json')
    try: template_history = json.load(open(template_history_path, 'r'))
    except Exception: template_history = {}

    # msa config
    if msa_config is None: chain2msa_paths = {}
    else: chain2msa_paths = json.load(open(msa_config, 'r'))

    best_config, best_scrmsd = None, 1e10
    for i, config_name in enumerate(sorted_history):
        if i == topk: break
        # run alphafold3
        path = history_configs[config_name]
        boltz_res_dir = os.path.join(root_dir, 'results', *config_name.split('_'))  # root_dir/results/roundx/n
        af3_out_dir = os.path.join(boltz_res_dir, 'af3_rectification')
        af3_model_path = os.path.join(af3_out_dir, 'output', 'AF3', 'AF3_model.cif')
        if not os.path.exists(af3_model_path):  # not already predicted in previous rounds
            if os.path.exists(af3_out_dir): shutil.rmtree(af3_out_dir)  # maybe leftovers from previous breakdown
            os.makedirs(af3_out_dir)
            _run_af3(path, template_dir, template_history, chain2msa_paths, af3_out_dir)
        if not os.path.exists(af3_model_path): # failed
            print_log(f'AF3 prediction of {config_name} failed')
            continue
        # get scRMSD
        ref_path = os.path.join(boltz_res_dir, config_name + '_model_0.cif')
        tgt_chains, lig_chains = [], []
        config = yaml.safe_load(open(path, 'r'))
        for item in config['sequences']:
            if 'protein' not in item: continue  # TODO: what about other kind of targets
            chain_id = item['protein']['id']
            if chain_id not in config['generator']['masks']: tgt_chains.append(chain_id)
            else: lig_chains.append(chain_id)
        sc_rmsd, _ = get_scRMSD(ref_path, af3_model_path, tgt_chains, lig_chains)
        if sc_rmsd < best_scrmsd: best_config, best_scrmsd = config_name, sc_rmsd
    
    json.dump(template_history, open(template_history_path, 'w'), indent=2)
    # return the config name of the best one
    return best_config, best_scrmsd
