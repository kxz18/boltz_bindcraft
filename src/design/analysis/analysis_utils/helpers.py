#!/usr/bin/python
# -*- coding:utf-8 -*-

def get_metric_val(d, name):
    if name == 'FWAway_dist': return d['FWAway'][0]
    elif name == 'FWAway_cnt': return d['FWAway'][1]
    if name == 'FWAwayExp_dist': return d['FWAwayExp'][0]
    elif name == 'FWAwayExp_cnt': return d['FWAwayExp'][1]
    return d[name]


def flatten_metric_names(d):
    names = []
    for name in d:
        if 'FWAway' in name: names.extend([(name + '_dist'), (name + '_cnt')])
        else: names.append(name)
    return names


def get_topk_val(vals, name, topk=1):
    name = name.lower()
    if 'plddt' in name: reverse = True
    elif 'ptm' in name: reverse = True
    elif 'rmsd' in name: reverse = False
    elif name == 'epitope': reverse = False
    elif 'fwaway' in name:
        if 'dist' in name: reverse = True
        elif 'cnt' in name: reverse = False
    elif name == 'total': reverse = False
    elif 'ranking_score' in name: reverse = True
    elif 'ipae' in name: reverse = False
    else: raise NotImplementedError(f'ranking method for {name} not implemented')
    return sorted(vals, reverse=reverse)[:topk]