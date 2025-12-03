#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import yaml
import time
import shutil
import argparse
from dataclasses import dataclass

import ray

from utils.logger import print_log

from .constructor import construct_input_json


@dataclass
class Task:
    id: str
    input_json: str
    log_dir: str
    output_dir: str
    status_file: str
    exit_code: int = 0
    elapsed_time: int = 0
    model: str = 'AF3'


def af3_cmd(gpu_ids, input_path, out_dir):
    cmd = f'''
        CUDA_VISIBLE_DEVICES={','.join(gpu_ids)} \
        bash /work/lpdi/users/xkong/codes/alphafold3/alphafold3_predict.sh \
        --json_path {input_path} \
        --output_dir {out_dir} 2>&1
    '''
    return cmd


def is_input_file_ext(filename, model):
    mapping = {
        'AF3': '.json',
        'Protenix': '.json',
        'Protenix-Mini': '.json',
        'Boltz2': '.yaml',
        'Boltz2-Force': '.yaml',
        'AFM': '.a3m'
    }
    return filename.endswith(mapping[model])


def get_task_name(path, model):
    if model == 'AF3': return json.load(open(path, 'r'))['name']
    elif model == 'AFM': return os.path.splitext(os.path.basename(path))[0]
    elif (model == 'Protenix') or (model == 'Protenix-Mini'): return json.load(open(path, 'r'))[0]['name']
    elif model.startswith('Boltz2'): return os.path.splitext(os.path.basename(path))[0]
    else: raise NotImplementedError(f'model {model} get_task_name not implemented')


@ray.remote(num_cpus=4, num_gpus=1)
def run(task: Task):
    start = time.time()
    gpu_ids = [str(i) for i in ray.get_gpu_ids()]
    input_json = os.path.abspath(task.input_json)
    out_dir = os.path.abspath(task.output_dir)
    if task.model == 'AF3': cmd = af3_cmd(gpu_ids, input_json, out_dir)
    else: raise NotImplementedError(f'model type {task.model} not implemented')
    try:
        with open(task.status_file, 'w') as fout: fout.write('PROCESSING\n')
        p = os.popen(cmd)
        print_log(f'command initiated: {cmd}')
        sys.stdout.flush()
        text = p.read()
        status = p.close()
        if status: status = os.waitstatus_to_exitcode(status)
        else: status = 0
        name = get_task_name(input_json, task.model)
        log_out = os.path.join(os.path.abspath(task.log_dir), name)
        os.makedirs(log_out, exist_ok=True)
        with open(os.path.join(log_out, 'log.txt'), 'w') as fout:
            fout.write(text)
        task.exit_code = status

        # add file marker
        marker = 'SUCCEEDED' if task.exit_code == 0 else 'FAILED'
        with open(task.status_file, 'w') as fout: fout.write(marker + '\n')
    except Exception as e:
        print_log(f'task {task.id} failed due to: {e}', level='ERROR')
        task.exit_code = 1
    # add time
    task.elapsed_time = time.time() - start
    return task


def form_task(json_path, model):
    parent_dir = os.path.dirname(json_path)
    name = get_task_name(json_path, model)
    status_file = os.path.join(parent_dir, 'logs', name, '_STATUS')
    if os.path.exists(status_file) and (open(status_file, 'r').read().rstrip() in ['SUCCEEDED', 'FAILED']): return None # already finished
    else:
        os.makedirs(os.path.join(parent_dir, 'logs', name), exist_ok=True)
        with open(status_file, 'w') as fout: fout.write('ADDED\n')
    os.makedirs(os.path.join(parent_dir, 'output'))
    task = run.remote(Task(
        id=json_path,
        input_json=json_path,
        log_dir=os.path.join(parent_dir, 'logs'),
        output_dir=os.path.join(parent_dir, 'output'),
        status_file=status_file,
        model=model
    ))
    print_log(f'task {json_path} added')
    return task


def recursive_scan(template_dir, template_history, dir, prefix, visited, tasks, max_queue_size=100, model='AF3'):
    if len(tasks) >= max_queue_size: return
    loss_traj = os.path.join(dir, 'loss_traj.json')
    if os.path.exists(loss_traj):
        print_log(f'getting tasks from {dir}', level='DEBUG')
        for n in os.listdir(dir):
            full_path = os.path.join(dir, n)
            if (full_path in visited) or os.path.isfile(full_path): continue
            af3_dir = os.path.join(full_path, 'af3')
            if os.path.exists(af3_dir):
                shutil.rmtree(af3_dir)
            os.makedirs(af3_dir, exist_ok=True)
            # get sequences
            config = os.path.join(dir, '..', '..', 'configs', prefix[-1] + f'_{n}.yaml') # roundx_n.yaml
            config = yaml.safe_load(open(config, 'r'))
            # form json input for AF3
            chain2template_cif = {}
            for item in config['templates']:
                for c1, c2 in zip(item['chain_id'], item['template_id']):
                    chain2template_cif[c1] = (c2, item['cif'])
            construct_input_json(
                chain2seqs={ item['protein']['id']: item['protein']['sequence'] for item in config['sequences'] },
                chain2template_cif=chain2template_cif,
                template_dir=template_dir,
                template_history=template_history,
                task_name='AF3',
                out_dir=af3_dir
            )
            # append task
            task = form_task(os.path.join(af3_dir, 'AF3.json'), model)
            if task is not None: tasks.append(task)
            visited[full_path] = True
            if len(tasks) >= max_queue_size: return
    elif os.path.isdir(dir):
        # loop all the subdirs
        for subdir in os.listdir(dir):
            print_log(f'Descending into directory {os.path.join(dir, subdir)}', level='DEBUG')
            recursive_scan(template_dir, template_history, os.path.join(dir, subdir), prefix + [subdir], visited, tasks, max_queue_size, model)
            if len(tasks) >= max_queue_size: return


def scan_tasks(dir, visited, model):
    template_dir = os.path.join(dir, 'af3_templates')
    os.makedirs(template_dir, exist_ok=True)
    template_history_path = os.path.join(template_dir, 'template_history.json')
    try: template_history = json.load(open(template_history_path, 'r'))
    except Exception: template_history = {}

    tasks = []
    recursive_scan(
        template_dir=template_dir,
        template_history=template_history,
        dir=dir,
        prefix=[],
        visited=visited,
        tasks=tasks,
        model=model
    )
    json.dump(template_history, open(template_history_path, 'w'), indent=2)

    return tasks


def main(args):
    ray.init()
    print_log('Ray initialized')
    idling_start = None
    finish_cnt = 0
    try:
        futures, visited = [], {}
        while True:
            print_log(f'Start scanning {args.task_dir}')
            tasks = scan_tasks(args.task_dir, visited, args.model)
            # for task in tasks: futures.append(run.remote(task))
            futures.extend(tasks)
            if len(tasks) > 0:
                print_log(f'Scanned {args.task_dir} and {len(tasks)} tasks appended...')
                sys.stdout.flush()
            else: print_log('No new tasks identified.')
            print_log(f'{finish_cnt} tasks finished. {len(futures)} tasks running...')
            time.sleep(10)
            while len(futures) > 0:
                done_ids, futures = ray.wait(futures, num_returns=1, timeout=1)
                if len(done_ids) == 0: break    # not any tasks finished yet
                for done_id in done_ids:
                    task = ray.get(done_id)
                    print_log(f'{task.id} finished. Elapsed {round(task.elapsed_time, 2)}s. Exit code: {task.exit_code}.')
                    sys.stdout.flush()
                    finish_cnt += 1
            if len(futures) == 0:
                if idling_start is None: idling_start = time.time()
                idling_span = time.time() - idling_start
                print_log(f'Idling for {round(idling_span, 2)}s...')
                if idling_span > 600:
                    print_log('No task for too long. Stop server.')
                    break
            else: idling_start = None
            sys.stdout.flush()

    except KeyboardInterrupt:
        print_log('Stopping...')
    ray.shutdown()



def parse():
    parser = argparse.ArgumentParser(description='AF3 server')
    parser.add_argument('--task_dir', type=str, required=True, help='Directory to store tasks')
    parser.add_argument('--model', type=str, choices=['AF3'], default='AF3')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())