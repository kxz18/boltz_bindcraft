
demo to run structure prediction:

```bash
python -m design.main --config ../demo/HA.yaml --out_dir ../tmp/out_demo/ --ckpt_dir ../.boltz/
```

kernel codes changed:

1. `boltz/model/layers/pairformer.py` for checkpointing
2. `boltz/model/modules/trunkv2.py` MSA module for checkpointing


additional dependencies:

matplotlib, seaborn

```bash
python -m design.analysis.traj --res_dir ../tmp/HA-stem
```

ray, biotite

for af3 server

```bash
python -m af3.struct_server --task_dir ../../tmp/
```


distribution of metrics:

```bash
python -m design.analysis.metrics --res_dir ../tmp/HA-stem/HA-stem_0 --tgt_chains AB --lig_chains H
```