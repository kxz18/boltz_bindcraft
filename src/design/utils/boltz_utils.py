#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import platform
from dataclasses import asdict
from typing import Optional, Literal
from pathlib import Path

import torch
from rdkit import Chem
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy

from boltz.data.types import Manifest
from boltz.main import download_boltz2, check_inputs, process_inputs, filter_inputs_structure
from boltz.main import BoltzProcessedInput, PairformerArgsV2, MSAModuleArgs, Boltz2DiffusionParams, Boltz2InferenceDataModule, BoltzWriter, BoltzSteeringParams
from boltz.model.models.boltz2 import Boltz2

from .logger import print_log



def prepare_boltz2(
        data: str,
        out_dir: str,
        cache: str,

        checkpoint: Optional[str] = None,
        devices: int = 1,
        accelerator: str = "gpu",

        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        step_scale: Optional[float] = None,
        write_full_pae: bool = False,
        write_full_pde: bool = False,

        output_format: Literal["pdb", "mmcif"] = "mmcif",
        num_workers: int = 2,
        override: bool = False,
        seed: Optional[int] = None,

        use_msa_server: bool = False,
        msa_server_url: str = "https://api.colabfold.com",
        msa_pairing_strategy: str = "greedy",
        msa_server_username: Optional[str] = None,
        msa_server_password: Optional[str] = None,
        api_key_header: Optional[str] = None,
        api_key_value: Optional[str] = None,

        use_potentials: bool = False,
        method: Optional[str] = None,

        preprocessing_threads: int = 1,
        max_msa_seqs: int = 8192,
        subsample_msa: bool = True,
        num_subsampled_msa: int = 1024,
        no_kernels: bool = False,
        write_embeddings: bool = False,
    ):
    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        # Disable kernel tuning by default,
        # but do not modify envvar if already set by caller
        os.environ[key] = os.environ.get(key, "1")

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Get MSA server credentials from environment variables if not provided
    if use_msa_server:
        if msa_server_username is None:
            msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
        if msa_server_password is None:
            msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
        if api_key_value is None:
            api_key_value = os.environ.get("MSA_API_KEY_VALUE")
        
        print_log(f"MSA server enabled: {msa_server_url}")
        if api_key_value:
            print_log("MSA server authentication: using API key header")
        elif msa_server_username and msa_server_password:
            print_log("MSA server authentication: using basic auth")
        else:
            print_log("MSA server authentication: no credentials provided")

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download_boltz2(cache)

    # Validate inputs
    data = check_inputs(data)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        boltz2=True,
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
    )

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Filter out existing predictions
    filtered_manifest = filter_inputs_structure(
        manifest=manifest,
        outdir=out_dir,
        override=override,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )


    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        start_method = "fork" if platform.system() != "win32" and platform.system() != "Windows" else "spawn"
        strategy = DDPStrategy(start_method=start_method)
        if len(filtered_manifest.records) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions, taking the minimum."
            )
            print_log(msg)
            if isinstance(devices, list):
                devices = devices[: max(1, len(filtered_manifest.records))]
            else:
                devices = max(1, min(len(filtered_manifest.records), devices))

    # Set up model parameters
    diffusion_params = Boltz2DiffusionParams()
    step_scale = 1.5 if step_scale is None else step_scale
    diffusion_params.step_scale = step_scale
    pairformer_args = PairformerArgsV2()

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=True,
    )

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
        boltz2=True,
        write_embeddings=write_embeddings,
    )

    # Set up trainer
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision="bf16-mixed",
    )

    if filtered_manifest.records:
        msg = f"Running structure prediction for {len(filtered_manifest.records)} input"
        msg += "s." if len(filtered_manifest.records) > 1 else "."
        print_log(msg)

        # Create data module
        data_module = Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method=method,
        )

        # Load model
        if checkpoint is None: checkpoint = cache / "boltz2_conf.ckpt"

        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
        }

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = use_potentials
        steering_args.physical_guidance_update = use_potentials

        model_module = Boltz2.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not no_kernels,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        model_module.eval()

    return trainer, model_module, data_module