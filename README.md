# Randomized Smoothing with Contrastive Pretraining

> Improving certified adversarial robustness through self-supervised contrastive pretraining combined with randomized smoothing.

## Overview

This repository contains the implementation for research on using **contrastive self-supervised pretraining** (SimCLR, SupContrast) to improve the certified robustness of classifiers when combined with **randomized smoothing**. Randomized smoothing provides provable guarantees against adversarial perturbations by averaging predictions over Gaussian noise, and the quality of the base classifier's representations directly impacts the certified radius.

## Project Structure

```
├── smoothing/           # Core randomized smoothing implementation
│   ├── code/            # Certification and prediction scripts
│   └── experiments.MD   # Experiment configurations and results
├── smoothing-catrs/     # CAtRS (Certified Adversarial Training) variant
├── SupContrast/         # Supervised and self-supervised contrastive learning
│   ├── main_supcon.py              # Standard SupCon training
│   ├── main_supcon_hierarchy.py    # Hierarchical contrastive learning
│   ├── main_ce.py                  # Cross-entropy baseline
│   ├── main_linear.py              # Linear evaluation
│   └── losses.py                   # Contrastive loss implementations
└── notebooks/           # Analysis and visualization notebooks
```

## Methods

- **SupContrast** — supervised contrastive pretraining with hierarchy-aware loss variants
- **Randomized Smoothing** — Gaussian noise-based certified defense
- **CAtRS** — certified adversarial training extension
- **Linear Probing** — evaluating representation quality after pretraining

## Tech Stack

- Python, PyTorch
- Certified robustness evaluation
- CIFAR-10/100 and ImageNet benchmarks

## Related

- [SelfSupForRanSmooth](https://github.com/mMirmohammadi/SelfSupForRanSmooth) — companion mmpretrain fork with additional self-supervised methods
