# Baselines

Uniform/non-uniform CH4:O2 (1:2) control systems for validating stochastic
imbalance theory. **Generated, not committed** — run:

```bash
python ../scripts/03_generate_ch4.py          # normal-density splited baselines
```

Sizes N ∈ {1,2,4,8,16,32,64,128,256,512,1024} thousand atoms.

## Variants

| Variant | File | Description |
|---------|------|-------------|
| **splited** (default) | `ch4_Nk_splited.xyz` | Normal density: CH4 at x>0, O2 at x<0, physical spacing (min distance ~1.09 Å). Validates the stochastic I2 theory. |
| cluster (opt-in) | `ch4_Nk_cluster.xyz` | Extreme artificial non-uniformity for partition-imbalance stress: CH4 packed into a corner so atoms **overlap by design**. Generate with `--variants cluster`; not shipped, never feed to an energy model. |

Positions are bit-reproducible on any platform / NumPy version (see
`../scripts/README.md`).
