#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

import numpy as np
from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.Data.IUPACData import protein_letters_3to1
from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure
from biotite.structure import residue_iter

from .base import StructPredTask, ChainData, TemplateData


def align_sequences(sequence_A, sequence_B, **kwargs):
    """
    Performs a global pairwise alignment between two sequences
    using the BLOSUM62 matrix and the Needleman-Wunsch algorithm
    as implemented in Biopython. Returns the alignment, the sequence
    identity and the residue mapping between both original sequences.

    The choices of gap_open and gap_extend are domain conventions which
    relate to the usage of BLOSUM62
    """

    matrix = kwargs.get('matrix', substitution_matrices.load("BLOSUM62"))
    gap_open = kwargs.get('gap_open', -10.0)
    gap_extend = kwargs.get('gap_extend', -0.5)

    alns = pairwise2.align.globalds(sequence_A, sequence_B,
                                    matrix, gap_open, gap_extend,
                                    penalize_end_gaps=(False, False) )

    best_aln = alns[0]
    aligned_A, aligned_B, score, begin, end = best_aln

    return aligned_A, aligned_B


def extract_seq_from_biotite_atom_array(atom_array):
    seq, indices = [], []
    for i, residue in enumerate(residue_iter(atom_array)):
        if residue[0].hetero: continue
        resname = residue[0].res_name
        aa = protein_letters_3to1.get(resname.capitalize(), "X")
        if aa == 'X': continue
        seq.append(aa)
        indices.append(i)
    return "".join(seq), indices


def get_template(seq, c, cif_path, out_path):
    # load the original cif file
    file = CIFFile.read(cif_path)
    struct = get_structure(file, include_bonds=True, extra_fields=['atom_id', 'b_factor', 'occupancy'])[0]
    struct = struct[struct.res_name != 'HOH']   # get rid of water
    
    file = CIFFile()
    chain_struct = struct[struct.chain_id == c].copy()
    unique_res_ids = np.unique(chain_struct.res_id)
    res_id_map = {old: new for new, old in enumerate(unique_res_ids, start=1)}
    chain_struct.res_id = np.array([res_id_map[r] for r in chain_struct.res_id])
    set_structure(file, chain_struct, include_bonds=True)
    file.write(out_path)
    struct_seq, struct_indices = extract_seq_from_biotite_atom_array(chain_struct)

    aligned_seq, aligned_struct_seq = align_sequences(seq, struct_seq)
    query_idxs, template_idxs = [], []
    j = 0
    for i, (a, b) in enumerate(zip(aligned_seq, aligned_struct_seq)):
        if (a != '-') and (b != '-'):
            query_idxs.append(i)
            template_idxs.append(struct_indices[j])
        if b != '-': j += 1
    # add release date for alphafold
    with open(out_path, 'a+') as fout: fout.write('_pdbx_audit_revision_history.revision_date 2012-12-19\n#')
    return TemplateData(
        cif_path=out_path,
        query_idxs=query_idxs,
        template_idxs=template_idxs
    )


def iterable_equal(l1, l2):
    if len(l1) != len(l2): return False
    equal = True
    for a, b in zip(l1, l2):
        if a != b:
            equal = False
            break
    return equal


def construct_input_json(chain2seqs, chain2msa_paths, chain2template_cif, template_dir, template_history, task_name, out_dir, n_seeds=1):

    chains = []
    for c in chain2seqs:
        seq = chain2seqs[c]
        if c not in chain2template_cif: # no template specified
            template_data = []
        elif (seq in template_history) and iterable_equal(chain2template_cif[c], template_history[seq][0]):
            _, template_file, query_idxs, template_idxs = template_history[seq]
            template_data = [TemplateData(
                cif_path=template_file,
                query_idxs=query_idxs,
                template_idxs=template_idxs
            )]
        else:
            out_file = os.path.join(template_dir, f't{len(template_history)}.cif')    # add tmp for colabfold AFM which require four-letter name for template
            template_data = [get_template(seq, chain2template_cif[c][0], chain2template_cif[c][1], out_file)]   # seq, chain in template, cif, out file
            template_history[seq] = (chain2template_cif[c], template_data[0].cif_path, template_data[0].query_idxs, template_data[0].template_idxs)
        chains.append(ChainData(
            id=c,
            sequence=seq,
            msa_path=chain2msa_paths.get(c, ''),    # stop msa
            templates=template_data
        ))
    
    task = StructPredTask(
        name=task_name,
        chains=chains,
        props={}
    )

    task.to_af3_json(out_dir, n_seeds=n_seeds)