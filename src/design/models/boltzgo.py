#!/usr/bin/python
# -*- coding:utf-8 -*-
import gc
import time
from dataclasses import dataclass
from typing import Any, Optional, Literal

import yaml
import torch
from torch import Tensor, nn

from boltz.model.models.boltz2 import Boltz2
from boltz.model.modules.trunkv2 import InputEmbedder
from boltz.data.const import tokens, prot_token_to_letter
from boltz.model.modules.diffusionv2 import *

from design.utils.logger import print_log, cuda_memory_summary
from design.utils.seed import get_rng_state, set_rng_state, setup_seed
from .loss import parse_losses
from .info import ComplexInfo


def get_detached_tensors(tensors):
    detached = []
    for t in tensors: detached.append(t.detach().requires_grad_(True))
    return detached


def expand_like(src, tgt):
    src = src.reshape(*src.shape, *[1 for _ in tgt.shape[len(src.shape):]]) # [..., 1, 1, ...]
    return src.expand_as(tgt)


def normalize_prob(p, m):
    m = expand_like(m.to(p.device), p)
    p = torch.where(m, torch.softmax(p, dim=-1), p)
    return p


def logits_to_types(logits, generate_mask, k=1, sample_method='multinomial'):
    mask = [(0 if ((name not in prot_token_to_letter) or (name == 'UNK') or (name == '-')) else 1) for name in tokens]
    mask = torch.tensor(mask, dtype=bool, device=logits.device)
    generate_mask = generate_mask.to(logits.device)
    logits = logits.masked_fill((~mask[None, None, :]) & generate_mask.unsqueeze(-1), float('-inf'))
    if sample_method == 'multinomial':
        logits[generate_mask] = torch.softmax(logits[generate_mask], dim=-1)
        index = torch.multinomial(logits[0], num_samples=k, replacement=True)    # [N, k], assume batch size = 1
        all_seqs = []
        for j in range(k):
            all_seqs.append([prot_token_to_letter[tokens[i]] for i in index[:, j]])
    elif sample_method == 'argmax':
        index = torch.argmax(logits[0], dim=-1)
        all_seqs = [[prot_token_to_letter[tokens[i]] for i in index]]
    else:
        raise NotImplementedError(f'sample method {sample_method} not implemented')
    return all_seqs
    # return [prot_token_to_letter[tokens[i]] for i in index[0]]


class InputEmbedderWrapper(nn.Module):
    def __init__(
            self,
            input_embedder: InputEmbedder,
            atom_embedding_mode: Literal['common', 'none', 'unk'] = 'none',
        ):
        super().__init__()
        self.input_embedder = input_embedder
        self.atom_embedding_mode = atom_embedding_mode

    def forward(self, feats: dict[str, Tensor], affinity: bool = False) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The embedded tokens.

        """
        # Load relevant features
        # res_type = feats["res_type"].float()
        res_type = feats["res_type"]
        if affinity:
            profile = feats["profile_affinity"]
            deletion_mean = feats["deletion_mean_affinity"].unsqueeze(-1)
        else:
            profile = feats["profile"]
            deletion_mean = feats["deletion_mean"].unsqueeze(-1)

        # Compute input embedding
        embedder = self.input_embedder
        if self.atom_embedding_mode == 'none':
            print_log(f'Setting atom features to zero', level='DEBUG')
            a = 0
        elif self.atom_embedding_mode == 'common':
            q, c, p, to_keys = embedder.atom_encoder(feats)
            atom_enc_bias = embedder.atom_enc_proj_z(p)
            a, _, _, _ = embedder.atom_attention_encoder(
                feats=feats,
                q=q,
                c=c,
                atom_enc_bias=atom_enc_bias,
                to_keys=to_keys,
            )
        else: raise NotImplementedError(f'atom embedding mode {self.atom_embedding_mode} not implemented')
        s = (
            a
            + embedder.res_type_encoding(res_type)
            + embedder.msa_profile_encoding(torch.cat([profile, deletion_mean], dim=-1))
        )

        # if embedder.add_method_conditioning:
        #     s = s + embedder.method_conditioning_init(feats["method_feature"])
        # if embedder.add_modified_flag:
        #     s = s + embedder.modified_conditioning_init(feats["modified"])
        # if embedder.add_cyclic_flag:
        #     cyclic = feats["cyclic_period"].clamp(max=1.0).unsqueeze(-1)
        #     s = s + embedder.cyclic_conditioning_init(cyclic)
        # if embedder.add_mol_type_feat:
        #     s = s + embedder.mol_type_conditioning_init(feats["mol_type"])

        return s


@dataclass
class BoltzGOConfig:
    lr: float = 1.0
    converge_patience: int = 5                          # how many steps not updating the history_best_topk list can we tolerate?
    max_inner_steps: int = 10
    max_outer_steps: int = 100
    inner_enc_recycling_steps: Optional[int] = None     # if none use the default in Boltz2 (default is 3)
    inner_diffusion_steps: Optional[int] = None         # if none, use the default in Boltz2
    maintain_logits: bool = False                       # whether to keep the logits between outer loops. if true, the optimizer and the logits will be maintained. not recommended to be true as it easily leads to stuck in local optima
    use_history_best: bool = False                      # whether to use history best as starters for each outer loops
    history_best_topk: int = 1                          # random sample from topk best in history for outer loops
    fix_inner_loop_seed: Optional[int] = None           # if None, do not fix seed
    inner_loop_best_as_output: bool = False

    init_scale: float = 6.0         # scale for the initial logits (one_hot * scale - offset for the softmax)
    init_offset: float = 3.0        # offset for the initial logits
    x_grad_scale: float = 1000.0    # gradient concerning structures will decay during backprop through diffusion, so we need to enlarge it
    final_grad_rescale_factor: float = 1.0 # g = g * factor. it's better to control the grad norm between 1e-1 and 1e-2

    sample_k: int = 5           # number of samples for discretization
    sample_method: str = 'multinomial'

    verbose: bool = False
    print_history_topk: int = 5

    def check_validity(self):
        # if self.use_history_best: assert not self.maintain_logits, f'use_history_best={self.use_history_best} is not compatible with maintain_logits={self.maintain_logits}'
        assert self.sample_method in ['multinomial', 'argmax'], f'sample_method {self.sample_method} not recognized.'
        if self.sample_method == 'argmax': assert self.sample_k == 1, f'sample_k should be 1 with sample_method={self.sample_method}'


class BoltzGO(Boltz2):  # boltz with gradient optimization

    def init(
        self,
        atom_embedding_mode: Literal['common', 'none', 'unk'] = 'common',
    ):
        # basics
        self.is_generation = True  # do structure prediction or generation
        self.input_embedder_wrapper = InputEmbedderWrapper(self.input_embedder, atom_embedding_mode)
        self.confidence_module.pairformer_stack.set_force_checkpointing()
        self.pairformer_module.set_force_checkpointing()
        self.use_kernels = False    # otherwise backward through pairformer will throw errors
        self.outer_loop_count = 0
        # dynamic recording variables
        self._traj = []
        

    def setup_config(
        self,
        yaml_path: str,
    ):# otherwise backward through pairformer will throw errors

        # generation-related
        with open(yaml_path, 'r') as fin: config = yaml.safe_load(fin)
        self.generator_config = BoltzGOConfig(**config['generator']['config'])
        self.generator_config.check_validity()
        self.losses = parse_losses(config['generator']['loss']) 

        # complex information
        # chain ids
        chain_ids = []
        chain_orders = {}
        # generate mask
        self.masks = []
        masks = config['generator']['masks']
        for i, seq in enumerate(config['sequences']):
            chain_id = seq['protein']['id']
            chain_ids.extend([chain_id for _ in seq['protein']['sequence']])
            chain_orders[chain_id] = i
            if chain_id not in masks: self.masks.extend([False for _ in seq['protein']['sequence']])
            else: self.masks.extend([(False if c == '0' else True) for c in masks[chain_id]])
        self.masks = torch.tensor(self.masks, dtype=bool, device=self.structure_module.device)[None, :] # batch size = 1
        self.cplx_info = ComplexInfo(chain_ids, chain_orders, self.masks)

    def increase_outer_loop(self):
        self.outer_loop_count += 1

    def set_mode_generation(self, is_generation: bool=True):
        self.is_generation = is_generation

    def enable_param_gradients(self, mode: bool=True):
        for name, p in self.named_parameters():
            p.requires_grad_(mode)

    def get_loss(self, dict_out: dict, feats: dict):
        loss, loss_details = 0, {}
        for name in self.losses:
            w, loss_cls = self.losses[name]
            l, v = loss_cls(dict_out, feats, self.cplx_info)
            loss += w * l
            loss_details[name] = v
        loss_details['total'] = round(loss.item(), 2)
        return loss, loss_details

    def _initialize(self, batch):
        if self.generator_config.maintain_logits:
            if self.outer_loop_count == 0: # the initial round
                print_log(f'Initialize maintained logits from Gaussian')
                res_type = batch['res_type'].detach().float().requires_grad_(True)
                res_type[self.masks] = torch.randn_like(res_type[self.masks]) * self.generator_config.init_offset # random initialization
                optimizer = torch.optim.AdamW([res_type], lr=self.generator_config.lr)
                self._res_type, self._optimizer = res_type, optimizer
            else: res_type, optimizer = self._res_type, self._optimizer
        else:   # start from the discretized state
            res_type = batch['res_type'].detach().float().requires_grad_(True)
            with torch.no_grad():
                if self.outer_loop_count == 0:  # the initial round
                    print_log(f'Initialize residue logits from Gaussian')
                    res_type[self.masks] = torch.randn_like(res_type[self.masks]) * self.generator_config.init_offset # random initialization
                else:
                    res_type[self.masks] = res_type[self.masks] * self.generator_config.init_scale - self.generator_config.init_offset # project one-hot to larger scale as logits
            optimizer = torch.optim.AdamW([res_type], lr=self.generator_config.lr)
        return res_type, optimizer  # logits and optimizer

    def _diffusion_one_step(
        self,
        # step info
        step_idx,
        sigma_tm,
        sigma_t,
        gamma, 
        # inputs
        atom_coords,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        # randomness
        random_aug=None,
        eps=None,
        # local variables
        scaled_guidance_update=None,
        potentials=None,
        energy_traj=None,
        resample_weights=None,
        **network_condition_kwargs, # s_trunk, s_inputs, feats (only variables without grads will be used), diffusion_conditioning
                                    # diffusion conditioning: q, c, to_keys(a function), atom_enc_bias, atom_dec_bias, token_trans_bias
    ):
        
        step_scale = self.structure_module.step_scale
        shape = (*atom_mask.shape, 3)

        if random_aug is None:
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
        else: random_R, random_tr = random_aug
        atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
        atom_coords = (
            torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
        )
        # if atom_coords_denoised is not None:
        #     atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
        #     atom_coords_denoised = (
        #         torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
        #         + random_tr
        #     )
        if (not self.is_generation) and (
            steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ) and scaled_guidance_update is not None:
            scaled_guidance_update = torch.einsum(
                "bmd,bds->bms", scaled_guidance_update, random_R
            )

        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        t_hat = sigma_tm * (1 + gamma)
        steering_t = 1.0 - (step_idx / num_sampling_steps)
        noise_var = self.structure_module.noise_scale**2 * (t_hat**2 - sigma_tm**2)
        if eps is None: eps = sqrt(noise_var) * torch.randn(shape, device=self.structure_module.device)
        atom_coords_noisy = atom_coords + eps

        # with torch.no_grad():
        atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
        sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
        sample_ids_chunks = sample_ids.chunk(
            multiplicity % max_parallel_samples + 1
        )

        for sample_ids_chunk in sample_ids_chunks:
            atom_coords_denoised_chunk = self.structure_module.preconditioned_network_forward(
                atom_coords_noisy[sample_ids_chunk],
                t_hat,
                network_condition_kwargs=dict(
                    multiplicity=sample_ids_chunk.numel(),
                    **network_condition_kwargs,
                ),
            )
            atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk

        if (not self.is_generation) and steering_args["fk_steering"] and (
            (
                step_idx % steering_args["fk_resampling_interval"] == 0
                and noise_var > 0
            )
            or step_idx == num_sampling_steps - 1
        ):
            # Compute energy of x_0 prediction
            energy = torch.zeros(multiplicity, device=self.structure_module.device)
            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if parameters["resampling_weight"] > 0:
                    component_energy = potential.compute(
                        atom_coords_denoised,
                        network_condition_kwargs["feats"],
                        parameters,
                    )
                    energy += parameters["resampling_weight"] * component_energy
            energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

            # Compute log G values
            if step_idx == 0:
                log_G = -1 * energy
            else:
                log_G = energy_traj[:, -2] - energy_traj[:, -1]

            # Compute ll difference between guided and unguided transition distribution
            if (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
            ) and noise_var > 0:
                ll_difference = (
                    eps**2 - (eps + scaled_guidance_update) ** 2
                ).sum(dim=(-1, -2)) / (2 * noise_var)
            else:
                ll_difference = torch.zeros_like(energy)

            # Compute resampling weights
            resample_weights = F.softmax(
                (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                    -1, steering_args["num_particles"]
                ),
                dim=1,
            )

        # Compute guidance update to x_0 prediction
        if (not self.is_generation) and (
            steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ) and step_idx < num_sampling_steps - 1:
            guidance_update = torch.zeros_like(atom_coords_denoised)
            for guidance_step in range(steering_args["num_gd_steps"]):
                energy_gradient = torch.zeros_like(atom_coords_denoised)
                for potential in potentials:
                    parameters = potential.compute_parameters(steering_t)
                    if (
                        parameters["guidance_weight"] > 0
                        and (guidance_step) % parameters["guidance_interval"]
                        == 0
                    ):
                        energy_gradient += parameters[
                            "guidance_weight"
                        ] * potential.compute_gradient(
                            atom_coords_denoised + guidance_update,
                            network_condition_kwargs["feats"],
                            parameters,
                        )
                guidance_update -= energy_gradient
            atom_coords_denoised += guidance_update
            scaled_guidance_update = (
                guidance_update
                * -1
                * self.structure_module.step_scale
                * (sigma_t - t_hat)
                / t_hat
            )

        if (not self.is_generation) and steering_args["fk_steering"] and (
            (
                step_idx % steering_args["fk_resampling_interval"] == 0
                and noise_var > 0
            )
            or step_idx == num_sampling_steps - 1
        ):
            resample_indices = (
                torch.multinomial(
                    resample_weights,
                    resample_weights.shape[1]
                    if step_idx < num_sampling_steps - 1
                    else 1,
                    replacement=True,
                )
                + resample_weights.shape[1]
                * torch.arange(
                    resample_weights.shape[0], device=resample_weights.device
                ).unsqueeze(-1)
            ).flatten()

            atom_coords = atom_coords[resample_indices]
            atom_coords_noisy = atom_coords_noisy[resample_indices]
            atom_mask = atom_mask[resample_indices]
            if atom_coords_denoised is not None:
                atom_coords_denoised = atom_coords_denoised[resample_indices]
            energy_traj = energy_traj[resample_indices]
            if (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
            ):
                scaled_guidance_update = scaled_guidance_update[
                    resample_indices
                ]
            if token_repr is not None:
                token_repr = token_repr[resample_indices]

        if self.structure_module.alignment_reverse_diff:
            with torch.autocast("cuda", enabled=False):
                atom_coords_noisy = weighted_rigid_align(
                    atom_coords_noisy.float(),
                    atom_coords_denoised.float(),
                    atom_mask.float(),
                    atom_mask.float(),
                )

            atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
        atom_coords_next = (
            atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
        )

        atom_coords = atom_coords_next
        return atom_coords, (random_R, random_tr), eps, (scaled_guidance_update, energy_traj, resample_weights)

    def _diffusion_sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        **network_condition_kwargs,
    ):
        native_self = self
        self = self.structure_module    # hack the codes

        if steering_args is not None and (
            steering_args["fk_steering"]
            or steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            potentials = get_potentials(steering_args, boltz2=True)
        else: potentials = None

        if steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        else:
            energy_traj, resample_weights = None, None
        if (not native_self.is_generation) and (
            steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
        else: scaled_guidance_update = None
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        
        # record
        native_self._traj = [[atom_coords, None, None]]

        token_repr = None
        # atom_coords_denoised = None

        # gradually denoise
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            atom_coords, random_aug, eps, (scaled_guidance_update, energy_traj, resample_weights) = native_self._diffusion_one_step(
                step_idx,
                sigma_tm,
                sigma_t,
                gamma, 
                # inputs
                atom_coords,
                atom_mask,
                num_sampling_steps,
                multiplicity,
                max_parallel_samples,
                steering_args,
                # randomness
                random_aug=None,
                eps=None,
                # local variables
                scaled_guidance_update=scaled_guidance_update,
                potentials=potentials,
                energy_traj=energy_traj,
                resample_weights=resample_weights,
                **network_condition_kwargs,
            )
            native_self._traj[-1][-2] = random_aug
            native_self._traj[-1][-1] = eps
            native_self._traj.append([atom_coords, None, None])

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def _diffusion_sample_backward(
        self,
        atom_mask,
        dx,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        **network_condition_kwargs,
    ):
        native_self = self
        self = self.structure_module    # hack the codes

        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        s, s_inputs = get_detached_tensors([network_condition_kwargs['s_trunk'], network_condition_kwargs['s_inputs']])
        q, c, atom_enc_bias, atom_dec_bias, token_trans_bias = get_detached_tensors([network_condition_kwargs['diffusion_conditioning'][k] for k in ['q', 'c', 'atom_enc_bias', 'atom_dec_bias', 'token_trans_bias']])
        network_condition_kwargs['s_trunk'], network_condition_kwargs['s_inputs'] = s, s_inputs
        network_condition_kwargs['diffusion_conditioning']['q'] = q
        network_condition_kwargs['diffusion_conditioning']['c'] = c
        network_condition_kwargs['diffusion_conditioning']['atom_enc_bias'] = atom_enc_bias
        network_condition_kwargs['diffusion_conditioning']['atom_dec_bias'] = atom_dec_bias
        network_condition_kwargs['diffusion_conditioning']['token_trans_bias'] = token_trans_bias

        # gradients
        ds, ds_inputs, dq, dc, datom_enc_bias, datom_dec_bias, dtoken_trans_bias = (
            torch.zeros_like(x) for x in [s, s_inputs, q, c, atom_enc_bias, atom_dec_bias, token_trans_bias]
        )

        # backward
        with torch.set_grad_enabled(True):
            for reverse_step_idx, (sigma_tm, sigma_t, gamma) in enumerate(reversed(sigmas_and_gammas)):
                step_idx = len(sigmas_and_gammas) - reverse_step_idx - 1
                atom_coords, random_aug, eps = native_self._traj[step_idx]
                atom_coords = atom_coords.detach().requires_grad_(True)
                atom_coords_next, _, _, _ = native_self._diffusion_one_step(
                    step_idx,
                    sigma_tm,
                    sigma_t,
                    gamma, 
                    # inputs
                    atom_coords,
                    atom_mask,
                    num_sampling_steps,
                    multiplicity,
                    max_parallel_samples,
                    steering_args,
                    # randomness
                    random_aug=random_aug,
                    eps=eps,
                    # local variables
                    scaled_guidance_update=None,  # WARN: seems to have memory
                    **network_condition_kwargs,
                )
                dx, v_s_inputs, v_s, v_q, v_c, v_atom_enc_bias, v_atom_dec_bias, v_token_trans_bias = torch.autograd.grad(
                    outputs=atom_coords_next,
                    inputs=[atom_coords, s_inputs, s, q, c, atom_enc_bias, atom_dec_bias, token_trans_bias],
                    grad_outputs=dx,
                    retain_graph=False,
                    allow_unused=True,
                )
                with torch.no_grad():
                    if v_s_inputs is not None: ds_inputs += v_s_inputs
                    if v_s is not None: ds += v_s
                    if v_q is not None: dq += v_q
                    if v_c is not None: dc += v_c
                    if v_atom_enc_bias is not None: datom_enc_bias += v_atom_enc_bias
                    if v_atom_dec_bias is not None: datom_dec_bias += v_atom_dec_bias
                    if v_token_trans_bias is not None: dtoken_trans_bias += v_token_trans_bias
        native_self._traj = []  # cleanup
        return  ds, ds_inputs, dq, dc, datom_enc_bias, datom_dec_bias, dtoken_trans_bias 

    def _encode(
        self,      
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        in_backward: int = False
    ):
        s_inputs = self.input_embedder_wrapper(feats)   # using the wrapper
        # Initialize the sequence embeddings
        s_init = self.s_init(s_inputs)

        # Initialize pairwise embeddings
        z_init = (
            self.z_init_1(s_inputs)[:, :, None]
            + self.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        if self.bond_type_feature:  # default True
            z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)

        # Perform rounds of the pairwise stack
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # Compute pairwise mask
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        if self.run_trunk_and_structure:    # default True
            for i in range(recycling_steps + 1):
                # Apply recycling
                s = s_init + self.s_recycle(self.s_norm(s))
                z = z_init + self.z_recycle(self.z_norm(z))

                # Compute pairwise stack
                if self.use_templates:  # default True
                    if self.is_template_compiled and not self.training:
                        template_module = self.template_module._orig_mod  # noqa: SLF001
                    else:
                        template_module = self.template_module

                    # z = z + template_module(
                    #     z, feats, pair_mask, use_kernels=self.use_kernels, checkpoint=in_backward
                    # )
                    z = z + torch.utils.checkpoint.checkpoint(
                        template_module, z, feats, pair_mask, self.use_kernels, use_reentrant=False
                    )
                if self.is_msa_compiled and not self.training:
                    msa_module = self.msa_module._orig_mod  # noqa: SLF001
                else:   # default this branch
                    msa_module = self.msa_module

                z = z + msa_module(
                    z, s_inputs, feats, use_kernels=self.use_kernels, checkpoint=in_backward
                )

                # Revert to uncompiled version for validation
                if self.is_pairformer_compiled and not self.training:
                    pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                else:   # default this branch
                    pairformer_module = self.pairformer_module
                # s, z = torch.utils.checkpoint.checkpoint(
                #     pairformer_module, s, z, mask, pair_mask, self.use_kernels, use_reentrant=False
                # )
                s, z = pairformer_module(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                    use_kernels=self.use_kernels,
                )

        pdistogram = self.distogram_module(z)
        dict_out = {
            "pdistogram": pdistogram,
            "s": s,
            "z": z,
        }

        if (
            self.run_trunk_and_structure
            and ((not self.training) or self.confidence_prediction)
            and (not self.skip_run_structure)
        ):  # default True
            q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                self.diffusion_conditioning(
                    s_trunk=s,
                    z_trunk=z,
                    relative_position_encoding=relative_position_encoding,
                    feats=feats,
                )
            )
            diffusion_conditioning = {
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            }
        return s_inputs, dict_out, diffusion_conditioning

    def _confidence(
            self,
            s_inputs,   # [1, N, 384]
            s,          # [1, N, 384]
            z,          # [1, N, N, 128]
            pdistogram, # [1, N, N, 1, 64]
            atom_coords,# [1, M, 3]
            feats: dict,
            diffusion_samples: int = 1,
            run_confidence_sequentially: bool = False,
        ) -> dict:
        return self.confidence_module(
            s_inputs=s_inputs,
            s=s,
            z=z,
            x_pred=(
                atom_coords
                if not self.skip_run_structure
                else feats["coords"].repeat_interleave(diffusion_samples, 0)
            ),
            feats=feats,
            pred_distogram_logits=(
                pdistogram[
                    :, :, :, 0
                ]  # TODO only implemented for 1 distogram
            ),
            multiplicity=diffusion_samples,
            run_sequentially=run_confidence_sequentially,
            use_kernels=self.use_kernels,
        )
    
    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = False,
    ) -> dict[str, Tensor]:

        feats['res_type'] = feats['res_type'].float()
        for k, v in feats.items():
            if isinstance(v, torch.Tensor) and (v.dtype.is_floating_point or v.is_complex()):
                v.requires_grad_(True)
        res_type = feats['res_type']    # this is the logits of residues
        with torch.no_grad():
            s_inputs, dict_out, diffusion_conditioning = self._encode(feats, recycling_steps)
            s, z, pdistogram = dict_out['s'], dict_out['z'], dict_out['pdistogram']
            with torch.autocast("cuda", enabled=False):
                struct_out = self._diffusion_sample(
                    s_trunk=s.float(),
                    s_inputs=s_inputs.float(),
                    feats=feats,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"].float(),
                    multiplicity=diffusion_samples,
                    max_parallel_samples=max_parallel_samples,
                    steering_args=self.steering_args,
                    diffusion_conditioning=diffusion_conditioning,
                )
                dict_out.update(struct_out)
        
        if not self.is_generation:
            with torch.no_grad():
                dict_out.update(self._confidence(s_inputs, s, z, pdistogram, dict_out['sample_atom_coords'], feats, diffusion_samples, run_confidence_sequentially))
                loss, loss_details = self.get_loss(dict_out, feats)
                dict_out['loss_details'] = loss_details
            return dict_out, {}

        atom_coords, s_inputs, s, z, pdistogram, res_type = get_detached_tensors([dict_out['sample_atom_coords'], s_inputs, s, z, pdistogram, res_type])
        feats['res_type'] = res_type
        with torch.set_grad_enabled(True):
            dict_out.update(self._confidence(s_inputs, s, z, pdistogram, atom_coords, feats, diffusion_samples, run_confidence_sequentially))
            dict_out['sample_atom_coords'] = atom_coords
            loss, loss_details = self.get_loss(dict_out, feats)
            dx, ds_inputs, ds, dz, dd, dres_type = torch.autograd.grad(loss, [atom_coords, s_inputs, s, z, pdistogram, res_type], retain_graph=False, allow_unused=True)
            # TODO: problem: why no grad to dx? x is discretized into pdistograms which are later added to z
        
        # backward for diffusion
        if dx is None: dx = torch.zeros_like(atom_coords)   # no coordinate related part in the loss
        if ds is None: ds = torch.zeros_like(s)
        if ds_inputs is None: ds_inputs = torch.zeros_like(s_inputs)
        vs, vs_inputs, dq, dc, datom_enc_bias, datom_dec_bias, dtoken_trans_bias = self._diffusion_sample_backward(
            dx=dx * self.generator_config.x_grad_scale,
            s_trunk=s.float(),
            s_inputs=s_inputs.float(),
            feats=feats,
            num_sampling_steps=num_sampling_steps,
            atom_mask=feats["atom_pad_mask"].float(),
            multiplicity=diffusion_samples,
            max_parallel_samples=max_parallel_samples,
            steering_args=self.steering_args,
            diffusion_conditioning=diffusion_conditioning,
        )

        if vs is not None: ds += vs
        if vs_inputs is not None: ds_inputs += ds_inputs

        # backward of encoder
        if dd is None: dd = torch.zeros_like(pdistogram)
        if dres_type is None: dres_type = torch.zeros_like(res_type)
        res_type = res_type.detach().requires_grad_(True)
        feats['res_type'] = res_type
        with torch.set_grad_enabled(True):
            s_inputs, tmp_dict_out, diffusion_conditioning = self._encode(feats, recycling_steps, in_backward=True)
            s, z, pdistogram = tmp_dict_out['s'], tmp_dict_out['z'], tmp_dict_out['pdistogram']
            dres_type += torch.autograd.grad(
                outputs=[s_inputs, s, z, pdistogram] + [diffusion_conditioning[k] for k in ['q', 'c', 'atom_enc_bias', 'atom_dec_bias', 'token_trans_bias']],
                inputs=res_type,
                grad_outputs=[ds_inputs, ds, dz, dd, dq, dc, datom_enc_bias, datom_dec_bias, dtoken_trans_bias],
            )[0]
        
        # set gradients
        dict_out['gradient'] = dres_type
        dict_out['loss_details'] = loss_details
        
        return dict_out, loss_details

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict:
        '''
            will call forward and process the results
        '''
        try:
            with torch.set_grad_enabled(False):
                if not self.is_generation:  # only do structure prediction
                    out, loss_details = self(
                        batch,
                        recycling_steps=self.predict_args["recycling_steps"],
                        num_sampling_steps=self.predict_args["sampling_steps"],
                        diffusion_samples=self.predict_args["diffusion_samples"],
                        max_parallel_samples=self.predict_args["max_parallel_samples"],
                        run_confidence_sequentially=True,
                    )
                else:   # do generation
                    step = 0
                    res_type, optimizer = self._initialize(batch)
                    if self.generator_config.verbose:
                        print_log(f'initialize, residues {res_type[self.masks][0]}')
                    loss_traj = []
                    best_loss, best_step, best_res_type = 1e10, None, None
                    while step < self.generator_config.max_inner_steps:
                        batch['res_type'] = normalize_prob(res_type.detach(), self.masks)
                        start = time.time()
                        if self.generator_config.fix_inner_loop_seed is not None:
                            rng_state = get_rng_state()
                            setup_seed(self.generator_config.fix_inner_loop_seed)
                        out, loss_details = self(
                            batch,
                            recycling_steps=default(self.generator_config.inner_enc_recycling_steps, self.predict_args["recycling_steps"]),
                            num_sampling_steps=default(self.generator_config.inner_diffusion_steps, self.predict_args["sampling_steps"]),
                            diffusion_samples=self.predict_args["diffusion_samples"],
                            max_parallel_samples=self.predict_args["max_parallel_samples"],
                            run_confidence_sequentially=True,
                        )
                        if self.generator_config.fix_inner_loop_seed is not None:
                            set_rng_state(rng_state)
                        loss_traj.append(loss_details)
                        with torch.set_grad_enabled(True):  # backward the normalize process
                            normalized_res_type = normalize_prob(res_type, self.masks)
                            res_type.grad = torch.autograd.grad(outputs=[normalized_res_type], inputs=res_type, grad_outputs=[out['gradient']])[0]
                        original_res_type = res_type.clone()
                        if self.generator_config.maintain_logits:
                            self._res_type.grad = res_type.grad
                            res_type = self._res_type
                        # record grad norm
                        grad_norm = torch.linalg.norm(res_type.grad[self.masks], dim=-1).mean().item()
                        grad_norm_str = '{:.2e}'.format(grad_norm)
                        res_type.grad = res_type.grad * self.generator_config.final_grad_rescale_factor
                        rescale_prompt = ' ({:.2e} after rescale)'.format(torch.linalg.norm(res_type.grad[self.masks], dim=-1).mean().item())
                        optimizer.step()
                        optimizer.zero_grad()
                        res_type[~self.masks] = original_res_type[~self.masks]
                        print_log(f'inner step {step}, total loss {loss_details["total"]}, grad norm {grad_norm_str}{rescale_prompt}, details {loss_details}, elapsed {round(time.time() - start, 2)}s')
                        if self.generator_config.verbose:
                            print_log(f'after updates, residues {res_type[self.masks][0]}, normalized {normalized_res_type[self.masks][0]}')
                            print_log(f'after updates, self._res_tyoe {self._res_type[self.masks][0]}')
                        if loss_details['total'] < best_loss:
                            best_loss, best_step, best_res_type = loss_details['total'], step, res_type.clone()
                        step += 1
                    if self.generator_config.inner_loop_best_as_output:
                        print_log(f'Using logits from step {best_step} as output due to its lowest loss')
                        res_type = best_res_type

            pred_dict = {"exception": False}
            if "keys_dict_batch" in self.predict_args:
                for key in self.predict_args["keys_dict_batch"]:
                    pred_dict[key] = batch[key]

            pred_dict["masks"] = batch["atom_pad_mask"]
            pred_dict["token_masks"] = batch["token_pad_mask"]
            pred_dict["s"] = out["s"]
            pred_dict["z"] = out["z"]

            if "keys_dict_out" in self.predict_args:
                for key in self.predict_args["keys_dict_out"]:
                    pred_dict[key] = out[key]
            pred_dict["coords"] = out["sample_atom_coords"]
            if self.confidence_prediction:
                # pred_dict["confidence"] = out.get("ablation_confidence", None)
                pred_dict["pde"] = out["pde"]
                pred_dict["plddt"] = out["plddt"]
                pred_dict["confidence_score"] = (
                    4 * out["complex_plddt"]
                    + (
                        out["iptm"]
                        if not torch.allclose(
                            out["iptm"], torch.zeros_like(out["iptm"])
                        )
                        else out["ptm"]
                    )
                ) / 5

                pred_dict["complex_plddt"] = out["complex_plddt"]
                pred_dict["complex_iplddt"] = out["complex_iplddt"]
                pred_dict["complex_pde"] = out["complex_pde"]
                pred_dict["complex_ipde"] = out["complex_ipde"]
                if self.alpha_pae > 0:
                    pred_dict["pae"] = out["pae"]
                    pred_dict["ptm"] = out["ptm"]
                    pred_dict["iptm"] = out["iptm"]
                    pred_dict["ligand_iptm"] = out["ligand_iptm"]
                    pred_dict["protein_iptm"] = out["protein_iptm"]
                    pred_dict["pair_chains_iptm"] = out["pair_chains_iptm"]
            if self.affinity_prediction:
                pred_dict["affinity_pred_value"] = out["affinity_pred_value"]
                pred_dict["affinity_probability_binary"] = out[
                    "affinity_probability_binary"
                ]
                if self.affinity_ensemble:
                    pred_dict["affinity_pred_value1"] = out["affinity_pred_value1"]
                    pred_dict["affinity_probability_binary1"] = out[
                        "affinity_probability_binary1"
                    ]
                    pred_dict["affinity_pred_value2"] = out["affinity_pred_value2"]
                    pred_dict["affinity_probability_binary2"] = out[
                        "affinity_probability_binary2"
                    ]
            
            pred_dict['loss_details'] = out['loss_details']
            if self.is_generation:
                pred_dict['optimized_res_logits'] = res_type
                pred_dict['optimized_res_type'] = logits_to_types(res_type, self.masks, self.generator_config.sample_k, self.generator_config.sample_method)
                pred_dict['loss_traj'] = loss_traj
            return pred_dict

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise e