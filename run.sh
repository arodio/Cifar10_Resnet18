#!/bin/bash

module load conda/2021.11-python3.9
source activate virt_pytorch_conda
cd paper_experiments/cifar10
sh fedavg.sh
