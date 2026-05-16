# LTSM-GNN

This repository provides the source code and processed data for **LTSM-GNN**, a label-driven subspace learning framework for multi-type protein-protein interaction (PPI) prediction.

LTSM-GNN learns label-specific feature subspaces for different PPI types and aligns learned type dependencies with empirical label co-occurrence statistics to improve prediction performance, especially for rare interaction types.

## Repository contents

- `model-git.py`: model architecture of LTSM-GNN.
- `train-git.py`: training script.
- `gnn_test.py`: evaluation script.
- `gnn_test_bigger.py`: evaluation script for larger-scale settings.
- `data-git.py`: data loading and preprocessing utilities.
- `data.7z`: processed data used for model training and testing.

## Requirements

The recommended environment can be created with:

```bash
conda env create -f environment.yml
conda activate ppi_gnn_linux
```

The code was tested with Python 3.9, PyTorch 2.5.1, CUDA 11.8 and PyTorch Geometric 2.6.1.

## Data preparation

The processed data are provided in `data.7z`. Please extract the archive before running the code:

```
7z x data.7z
```

The original benchmark datasets are available from the following sources:

- SHS27k and SHS148k: http://yellowstone.cs.ucla.edu/~muhao/pipr/SHS_ppi_beta.zip
- STRING: https://string-db.org/cgi/download

### Data splits

The files in `train_valid_index_json/` store the fixed data splits used in the manuscript.
 For historical naming reasons, `valid_index` denotes the held-out evaluation/test subset used for reporting the results, while `train_index` denotes the training subset. The random, BFS and DFS split files correspond to the partition protocols reported in the manuscript.

## Training

After extracting the data, run:

```
python train-git.py
```

## Testing

To evaluate the trained model, run:

```
python gnn_test.py
```

For larger-scale evaluation, run:

```
python gnn_test_bigger.py
```

## Reproducibility

The repository contains the source code, processed data and scripts required to reproduce the main experiments reported in the manuscript. Experiments are conducted under random, BFS and DFS partition protocols following the benchmark settings used in previous PPI prediction studies. The version associated with the submitted manuscript is release `v1.0.0`.

## Citation

If you use this code or data, please cite the corresponding manuscript:

```
 Wang et al. Link Type Subspace Modeling with GNNs for Protein-Protein Interaction Prediction.
```

## License

This project is released for non-commercial use.
