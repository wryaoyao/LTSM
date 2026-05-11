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

The code is implemented in Python and PyTorch. The main dependencies include:

```bash
python >= 3.8
pytorch
numpy
scipy
scikit-learn
pandas
tqdm
````

A detailed dependency file will be provided in `requirements.txt`.

## Data preparation

The processed data are provided in `data.7z`. Please extract the archive before running the code:

```bash
7z x data.7z
```

The original benchmark datasets are available from the following sources:

* SHS27k and SHS148k: [http://yellowstone.cs.ucla.edu/~muhao/pipr/SHS_ppi_beta.zip](http://yellowstone.cs.ucla.edu/~muhao/pipr/SHS_ppi_beta.zip)
* STRING: [https://string-db.org/cgi/download](https://string-db.org/cgi/download)

## Training

After extracting the data, run:

```bash
python train-git.py
```

## Testing

To evaluate the trained model, run:

```bash
python gnn_test.py
```

For larger-scale evaluation, run:

```bash
python gnn_test_bigger.py
```

## Reproducibility

The repository contains the source code, processed data and scripts required to reproduce the main experiments reported in the manuscript. Experiments are conducted under random, BFS and DFS partition protocols following the benchmark settings used in previous PPI prediction studies.

## Citation

If you use this code or data, please cite the corresponding manuscript:

```text
Fu et al. Link Type Subspace Modeling with GNNs for Protein-Protein Interaction Prediction.
```
## License

This project is released for academic and non-commercial research use.

