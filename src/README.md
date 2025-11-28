
demo to run structure prediction:

```bash
python -m design.main --config ../demo/HA.yaml --out_dir ../tmp/out_demo/ --ckpt_dir ../.boltz/
```

kernel codes changed:

1. `boltz/model/layers/pairformer.py` for checkpointing
2. `boltz/model/modules/trunkv2.py` MSA module for checkpointing


additional dependencies:

matplotlib, seaborn