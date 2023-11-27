# Energy and Forces

This experiment aims to explore the question of can SMIRNOFF style force fields be fit purely to energies and forces
without any MM minimization currently employed by the current OpenFF fitting targets.

## Dataset

The dataset used for this experiment is the same as the one used for the [Espaloma 0.3.0](). It can be downloaded
directly from Zenodo:

```shell
mkdir -p outputs

curl -o outputs/8150601.zip https://zenodo.org/api/records/8150601/files-archive

unzip outputs/8150601.zip -d outputs/8150601
cd outputs/8150601

for f in *.tar.gz; do tar -zxvf "$f"; done
rm -r *.tar.gz
```

It needs to be converted into a Hugging Face dataset (i.e. `pyarrow` format) to be used in descent:

```shell
python 001-convert-espaloma-data.py
```

This should create a `outputs/data-raw` directory containing each converted dataset.

Each molecule should be pre-parameterized with the SMIRNOFF force field being trained. This is because `smee` force
field objects are created from Interchange objects rather than SMIRNOFF force field files.

```shell
python 002-parameterize.py
```

This should save the `smee` force field and topology objects for each molecule as a single `pytorch` blob as
`outputs/openff-2.1.0.pt` directory.

Finally, any molecules that could not be parameterized should be removed from the dataset:

```shell
python 003-cluster-and-filter.py
```

This script contains a parameter (enabled by default) to optionally cluster and pick a subset of conformers for each
molecule in`gen2-opt`. Some entries in `gen2-opt` have ~3000 very similar conformers as I believe the full optimization
trajectory was pulled.

If clustering is enabled, the filtered and clustered data will be saved to `outputs/data-clustered` respectively.
Otherwise, the filtered data will be saved to `outputs/data-filtered`.

## Training

The training script is `004-train.py`. It contains hard-coded parameters for which dataset to use (i.e. '-clustered'
or '-filtered'), the number of epochs, and fitting hyperparameters.

It uses `torch.multiprocessing` to vertically scale the training across CPUs on the current machine, but in principle
should also work with something like `torchrun` if running across multiple nodes.

As training as progressing, metrics should be logged to `outputs/runs/<timestamp>`. These can be monitored with
`tensorboard`:

```shell
tensorboard --logdir outputs/runs
```

## Analysis
