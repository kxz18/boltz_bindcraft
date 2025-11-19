
demo to run structure prediction:

```bash
python -m design.main --config ../tmp/input.yaml --out_dir ../tmp/out_design --ckpt_dir ../.boltz/
```

kernel codes changed:

1. `boltz/model/layers/pairformer.py` for checkpointing
2. `boltz/model/modules/trunkv2.py` MSA module for checkpointing