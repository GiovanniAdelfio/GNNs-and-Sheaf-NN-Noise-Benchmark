# GNNs and Sheaf Neural Network — Noise Benchmark

This repository contains my personal contributions to a benchmark study on the robustness of Graph Neural Networks (GNNs) and Sheaf Neural Networks under various noise conditions, evaluated on node classification tasks.

## Context

The full benchmarking framework (training loop, dataset loading, evaluation pipeline, baseline GNN models) is built on top of a private repository developed by a colleague. This repository contains only the components I implemented on top of that framework.

## Contents

| File | Description |
|---|---|
| `model/SheafNN.py` | Implementation of the Sheaf Neural Network model |
| `methods/SheafNNHelper.py` | Method helper that integrates SheafNN into the benchmark framework |
| `methods/Sheaf_graphcleaner.py` | Method helper that integrates SheafNN into the benchmark framework while allowing graphcleaner processing|
| `model/methods/SheafNNTrainer.py` | Trainer class for SheafNN |
| `model/methods/sheaf_graphcleaner.py` | Trainer class for SheafNN which uses graphcleaner before training |
| `util/laplacian_builder.py` | Utility to build the Sheaf Laplacian |
| `config.yaml` | Example configuration file |

## Model Overview

The Sheaf Neural Network implemented here follows the **Neural Sheaf Diffusion** formulation from Bodnar et al. (NeurIPS 2022). The architecture consists of:

1. An input MLP that projects node features into a hidden embedding space
2. A sheaf learner that computes restriction maps for each edge (either as learnable parameters or via a small MLP)
3. A diffusion process over K layers using the normalised Sheaf Laplacian
4. An output MLP that maps node embeddings to class probabilities

## Reference

> Bodnar et al., *Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs*, NeurIPS 2022

## Notes

- The base benchmarking framework is not included as it is currently private.
- To reproduce the full benchmark, access to the base repository is required.
