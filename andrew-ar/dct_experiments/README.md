# DCT + CAA Backdoor Detection Experiments

Results for the NeurIPS 2026 submission on blind backdoor detection via MLP Jacobian fingerprinting.

## Directory Structure

```
dct_experiments/
├── neurips_manuscript/          NeurIPS submission (main.tex + 10 figures + .sty)
├── report_final/                Earlier draft + figure generation script
├── results_dct/                 All experimental results (CSV + PT files)
│   ├── joint/                      Joint cross-experiment (CAA ⊥ DCT)
│   │   ├── caa_directions_vpi_32layers.npy    (32, 4096) CAA directions
│   │   ├── caa_dct_projection.csv             Main orthogonality table
│   │   ├── caa_discriminability_vpi.csv       Per-layer cos distance
│   │   └── caa_auroc_vpi.csv                  Per-layer CAA AUROC
│   ├── v_matrices/              LinearDCT V matrices (our build, matches Ryan's)
│   │   └── V_linear_l{4,8,12,16,20,24,28,31}_f64.pt  (4096, 64) each
│   ├── tierB/                   caa_validation.py outputs for all 5 attacks
│   ├── deep/                    Investigation sweeps
│   │   ├── per_layer_auroc_all_attacks.csv   5 attacks × 32 layers
│   │   ├── score_distributions.csv           Clean vs poison scores
│   │   ├── singular_values.csv               SVD spectra by layer
│   │   ├── clean_model_auroc.csv             Non-backdoored baseline
│   │   ├── clean_model_trigger_strip.csv     Trigger-strip ablation
│   │   ├── tsne_layer{4,30}.csv              t-SNE projections
│   │   └── separation_heatmap.csv
│   ├── investigation/           Hyperparameter sweeps
│   │   ├── rank_sweep.csv                    Rank 1 to 128
│   │   └── ntrain_sweep.csv                  5 to 400 clean docs
│   ├── poison_sweep/            7B Jailbreak poison rate sweep (300 runs)
│   ├── 7b_refusal/              7B Refusal (375 runs)
│   ├── 13b/                     13B both tasks (540 runs)
│   ├── 70b/                     70B Jailbreak (300 runs)
│   ├── causal/                  Retrain-with-filtered-corpus results
│   ├── cross_model/             Mistral-7B detection
│   └── adversarial/             Lambda sweep (0 to 10)
└── scripts/                     Experiment code
```

## Key Results

- **Orthogonality** (Fig 1): cosine(CAA, DCT) ∈ [-0.05, +0.03] at every layer
- **Scale** (Table 1): L2 blind AUROC 0.978 (7B) → 0.990 (13B) → 0.990 (70B) at 50% poison
- **Attack invariance** (Table 2): std < 0.004 across 5 attacks
- **Causal** (Fig 9): K=50 removal drops ASR 75% → 12% (90% precision)
- **Cross-model** (Table 4): Mistral-7B AUROC 0.982
- **Adversarial** (Table 5, Fig 10): AUROC floors at 0.72 while ASR = 100%
- **Clean model** (Table 3, Fig 5): AUROC 0.968 without any backdoor adapter

## Reproducibility

All numbers in the draft come from CSVs in `results_dct/`. Total: 1,515 runs across 3 model scales, 5 attacks, 2 tasks, 5 poison rates, 3 seeds.
