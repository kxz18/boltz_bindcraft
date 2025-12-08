#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import yaml
import random
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TemplateData:
    cif_path: str
    query_idxs: List[int]       # start from zero, used by AF3
    template_idxs: List[int]    # start from zero, used by AF3
    # query_chains: List[str]
    # template_chains: List[str]


@dataclass
class ChainData:
    id: str # chain id
    sequence: str
    
    msa_path: Optional[str] = None # if disabled, set to ''
    templates: Optional[List[TemplateData]] = None  # if disabled, set to []


def msa_change_query_seq(in_path, out_path, seq):
    with open(in_path, 'r') as fin: lines = fin.readlines()
    lines[1] = seq + '\n'
    with open(out_path, 'w') as fout: fout.writelines(lines)


@dataclass
class StructPredTask:
    name: str
    chains: List[ChainData]
    props: dict

    def to_af3_json(self, out_dir, suffix='', include_chains=None, n_seeds=1):
        if include_chains is None: include_chains = [c.id for c in self.chains]
        data = []
        msa_dir = os.path.join(out_dir, 'msas', self.name + suffix)
        os.makedirs(msa_dir, exist_ok=True)

        for chain in self.chains:
            if chain.id not in include_chains: continue
            info = {
                'id': chain.id,
                'sequence': chain.sequence
            }
            if chain.msa_path == '':
                info['unpairedMsa'], info['pairedMsa'] = '', ''
            elif chain.msa_path is not None:
                unpaired_path = os.path.abspath(os.path.join(msa_dir, f'unpaired_{chain.id}.a3m'))
                paired_path = os.path.abspath(os.path.join(msa_dir, f'paired_{chain.id}.a3m'))
                msa_change_query_seq(chain.msa_path[0], unpaired_path, chain.sequence)
                msa_change_query_seq(chain.msa_path[1], paired_path, chain.sequence)
                info['unpairedMsaPath'] = unpaired_path
                info['pairedMsaPath'] = paired_path
            if chain.templates is not None:
                if len(chain.templates) == 0: info['templates'] = []    # disable template
                else:
                    info['templates'] = [
                        {
                            'mmcifPath': os.path.abspath(temp.cif_path),
                            'queryIndices': temp.query_idxs,
                            'templateIndices': temp.template_idxs
                        } for temp in chain.templates
                    ]
            
            data.append({ 'protein': info })

        input_json = {
            'name': self.name + suffix,
            'dialect': 'alphafold3',
            'version': 2,
            'modelSeeds': [random.randint(0, 4294967295) for _ in range(n_seeds)],
            'bondedAtomPairs': None,
            'userCCD': None,
            'sequences': data
        }
        with open(os.path.join(out_dir, self.name + suffix + '.json'), 'w') as fout: json.dump(input_json, fout, indent=2)
        return self.name + suffix   # task name
    
    def to_afm_dirs(self, out_dir, suffix='', include_chains=None, n_seeds=1):
        if include_chains is None: include_chains = [c.id for c in self.chains]
        task_name = self.name + suffix
        template_dir = os.path.join(out_dir, f'{task_name}_templates')
        os.makedirs(template_dir, exist_ok=True)

        chain2seqs, chain2msas = {}, {}

        for chain in self.chains:
            if chain.id not in include_chains: continue
            chain2seqs[chain.id] = chain.sequence
            if chain.msa_path is not None: chain2msas[chain.id] = chain.msa_path
            else: chain2msas[chain.id] = ''
            if chain.templates is not None and len(chain.templates) != 0:
                for temp in chain.templates:
                    os.system(f'cp {temp.cif_path} {os.path.join(template_dir, os.path.basename(temp.cif_path))}') 

        # write a3m
        fout = open(os.path.join(out_dir, f'{task_name}.a3m'), 'w')
        fout.write(f'#{",".join([str(len(chain2seqs[c])) for c in chain2seqs])}\t{",".join(["1" for _ in chain2seqs])}\n')
        fout.write('>' + '\t'.join([c for c in chain2seqs]) + '\n')
        fout.write(''.join([chain2seqs[c] for c in chain2seqs]) + '\n')
        for c in chain2seqs:
            s = ''
            for c1 in chain2seqs:
                if c1 == c: s += chain2seqs[c1]
                else: s += '-' * len(chain2seqs[c1])
            fout.write(f'>{c}\n')
            fout.write(s + '\n')
        fout.close()
        return task_name

    def to_protenix_json(self, out_dir, suffix='', include_chains=None, n_seeds=1):
        if include_chains is None: include_chains = [c.id for c in self.chains]
        add_args = {}
        data = []
        for chain in self.chains:
            if chain.id not in include_chains: continue
            info = {
                'sequence': chain.sequence,
                'count': 1,
            }
            if chain.msa_path == '':
                add_args['--use_msa'] = 'false'
            elif chain.msa_path is not None:
                info['msa'] = { # WARN: may be some API mismatch, as Protenix requires two files: pairing.a3m and non_pairing.a3m
                    'precomputed_msa_dir': os.path.abspath(os.path.dirname(chain.msa_path)),
                    'pairing_db': 'uniref100'
                }
            data.append({
                'proteinChain': info
            })

        input_json = [{
            'name': self.name + suffix,
            'sequences': data
        }]
        add_args['--seeds'] = ','.join([str(random.randint(0, 4294967295)) for _ in range(n_seeds)])
        input_json[0]['add_args'] = add_args
        with open(os.path.join(out_dir, self.name + suffix + '.json'), 'w') as fout: json.dump(input_json, fout, indent=2)
        return self.name + suffix   # task name

    def to_boltz2_yaml(self, out_dir, suffix='', include_chains=None, n_seeds=1, force=False):
        if include_chains is None: include_chains = [c.id for c in self.chains]
        data, templates = [], []
        for chain in self.chains:
            if chain.id not in include_chains: continue
            info = {
                'id': chain.id,
                'sequence': chain.sequence,
            }
            if chain.msa_path == '':
                info['msa'] = 'empty'
            elif chain.msa_path is not None:
                info['msa'] = chain.msa_path
            data.append({
                'protein': info
            })
            if (chain.templates is not None) and (len(chain.templates) > 0):
                for temp in chain.templates:
                    templates.append({
                        'cif': os.path.abspath(temp.cif_path)
                    })
                    if force:
                        templates[-1]['force'] = True
                        templates[-1]['threshold'] = 2.0

        input_yaml = {
            'sequences': data,
        }
        if len(templates): input_yaml['templates'] = templates
        add_args = { '--seed': str(random.randint(0, 4294967295)) }
        add_args['--diffusion_samples'] = 5 * n_seeds
        input_yaml['add_args'] = add_args
        with open(os.path.join(out_dir, self.name + suffix + '.yaml'), 'w') as fout: yaml.dump(input_yaml, fout, default_flow_style=False)
        
        return self.name + suffix   # task name


@dataclass
class StructPredResult:
    name: str
    cif_path: str
    summary_path: str

    @classmethod
    def from_af3(cls, name: str, dir: str) -> "StructPredResult":
        dir = os.path.join(dir, name)
        cif_path = os.path.join(dir, f'{name}_model.cif')
        summary_path = os.path.join(dir, f'{name}_summary_confidences.json')
        return cls(
            name=name,
            cif_path=cif_path,
            summary_path=summary_path
        )

    @classmethod
    def from_afm(cls, name: str, dir: str) -> "StructPredResult":
        dir = os.path.join(dir, name)
        cif_paths = []
        for fname in os.listdir(dir):
            # if fname.endswith('.pdb') and 'rank_001' in fname: break
            if fname.endswith('.pdb'): cif_paths.append(os.path.join(dir, fname))
        # cif_path = os.path.join(dir, fname)
        # summary_path = os.path.join(dir, fname.replace('unrelaxed', 'scores').replace('.pdb', '.json'))
        summary_paths = [f.replace('unrelaxed', 'scores').replace('.pdb', '.json') for f in cif_paths]
        return cls(
            name=name,
            cif_path=cif_paths,
            summary_path=summary_paths
        )

    @classmethod
    def from_protenix(cls, name: str, dir: str) -> "StructPredResult":
        dir = os.path.join(dir, name)
        file2summary = {}
        for seed_dir in os.listdir(dir):
            if not seed_dir.startswith('seed'): continue
            pred_dir = os.path.join(dir, seed_dir, 'predictions')
            # loop through all summaries
            for f in os.listdir(pred_dir):
                if not f.endswith('.json'): continue
                n = f.rstrip('.json').split('_')[-1]
                file2summary[os.path.join(pred_dir, f'{name}_sample_{n}.cif')] = os.path.join(pred_dir, f)
        best_cif = max(file2summary.keys(), key=lambda path: json.load(open(file2summary[path], 'r'))['ranking_score'])
        return cls(
            name=name,
            cif_path=best_cif,
            summary_path=file2summary[best_cif]
        )
    
    @classmethod
    def from_boltz(cls, name: str, dir: str) -> "StructPredResult":
        dir = os.path.join(dir, f'boltz_results_{name}', 'predictions', name)
        cif_path = os.path.join(dir, f'{name}_model_0.cif') # already ranked by confidence
        summary_path = os.path.join(dir, f'confidence_{name}_model_0.json')
        return cls(
            name=name,
            cif_path=cif_path,
            summary_path=summary_path
        )
