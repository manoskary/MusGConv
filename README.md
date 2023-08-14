# MusGConv: Music Graph Convolutions 

This package contains implementation of the paper _Perception-Inspired Graph Convolution for Music Understanding Tasks_
submitted to AAAI 2024.

## Requirements

Please install the version of torch that is compatible with your machine preferably using conda.

Use the following command to install torch with conda:
```shell
conda create -n musgconv python=3.8
conda activate musgconv
conda install pytorch cudatoolkit=<your version of cuda> -c pytorch
```
Similarly, make sure to install [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)


To install the rest of the requirements with pip do:
```shell
cd Path/to/musgconv
pip install -r requirements.txt
```

Install as a package:
```shell
pip install -e .
```

## Running the Experiments

To run the experiments, you can use the following command:

#### Cadence Detection

```shell
python  ./experiments/cadet.py
```

#### Roman Numeral Analysis

```shell
python ./experiments/chord_prediction.py
```

#### Composer Classification
    
```shell
python ./experiments/composer_classification.py
```

#### Voice Separation

```shell
python ./experiments/voice_separation.py
```

#### Parametrization

To run the experiments with different parameters you can add arguments on the command line. For example, to run the experiments with a different number of hidden layers, you can use the following command: 

```shell
python ./experiments/cadet.py --n_layers 3
```

To see all the possible arguments, you can use the following command:

```shell
python ./experiments/cadet.py --help
```

All experiments are logged with WANDB so if you want to log the experiments you can add the following arguments:

```shell
python ./experiments/cadet.py --use_wandb --wandb_entity <your_wandb_entity>
```


## Aknowledgements

The code of this repository is based on the following repositories:
- [Cadet](https://github.com/manoskary/cadet)
- [SymRep](https://github.com/anusfoil/SymRep)
- [AugmentedNet](https://github.com/napulen/AugmentedNet)
- [vocsep_ijcai2023](https://github.com/manoskary/vocsep_ijcai2023)
- [ChordGNN](https://github.com/manoskary/chordgnn)
