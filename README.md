[![Python](https://img.shields.io/badge/-Python_3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Paper](http://img.shields.io/badge/paper-arxiv.2405.09224-B31B1B.svg)](https://arxiv.org/abs/2405.09224)
[![Conference](http://img.shields.io/badge/IJCAI-2024-4b44ce.svg)](https://ijcai24.org/ai-arts-creativity-special-track-accepted-papers/)


# MusGConv: Music-informed Graph Convolutions 

This package contains implementation of the paper _Perception-Inspired Graph Convolution for Music Understanding Tasks_
submitted to IJCAI 2024. We propose a novel graph convolutional layer that is inspired by the way we perceive and learn
pitch in music theory. We show that our layer outperforms SOTA graph convolutional layers in four music understanding tasks.
The tasks are cadence detection, roman numeral analysis, composer classification and voice separation.

## Requirements

Please install the version of torch that is compatible with your machine preferably using conda.

Use the following command to install torch with conda:
```shell
conda create -n musgconv python=3.9
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

We provide two main scripts to reproduce the experiments and tables of the paper.
The first script is `run_main_experiment.py` which compares SOTA architectures for each 
of our tasks to the same architecture where the graph convolutions are replaced with MusGConv.
The second script is `run_ablation_experiment.py` which compares the performance of MusGConv with different
configurations of the parameters of the MusGConv layer.

To run the **main experiments**, you can use the following command:

```shell
python ./experiments/run_main_experiment.py --gpus <number of gpus> --use_wandb --wandb_entity <your_wandb_entity>
```

To run the **ablation experiments**, you can use the following command:

```shell
python ./experiments/run_ablation_experiment.py --gpus <number of gpus> --use_wandb --wandb_entity <your_wandb_entity>
```

For our paper, we used WandB to log our experiments. 
To reproduce our results, you can simply provide your own WandB entity. The project name, groups, jobs and run 
names will be automatically generated to match the paper.
The results of the experiments will be logged in the same fashion as in the paper, so you can compare your results with ours.
If you don't have a WandB account, you can create one for free.
You can find our results here: https://wandb.ai/vocsep/MusGConv
If you don't want to use WandB, don't use the flag `--use_wandb`.

#### Using GPU for Training

To use GPU for training, you can use the flag `--gpus` and specify the number of GPUs you want to use.
In our experiments, we used a single GTX 1080 Ti GPU with 11GB of memory. If your GPU has less memory, you need to reduce the batch size.
However, in the experiments above the batch size is set to match the results the paper.

### Running Individual Experiments

To run the individual experiments, you can use the following commands:
To change the parameters of the experiments, you can use the arguments of the scripts, use `script_name.py --help` to see the arguments.

```shell

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

## Cite
```bibtex
@inproceedings{karystinaios2024musgconv,
  title={Perception-Inspired Graph Convolution for Music Understanding Tasks},
  author={Karystinaios, Emmanouil and Foscarin, Francesco and Widmer, Gerhard},
  booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2024}
}
```


## Aknowledgements

We would like to give credit to the following repositories that we used to build our codebase:
- [Cadet](https://github.com/manoskary/cadet)
- [SymRep](https://github.com/anusfoil/SymRep)
- [AugmentedNet](https://github.com/napulen/AugmentedNet)
- [vocsep_ijcai2023](https://github.com/manoskary/vocsep_ijcai2023)
- [ChordGNN](https://github.com/manoskary/chordgnn)
