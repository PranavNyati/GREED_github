This repository contains data and pre-trained models for the paper [GREED: A Neural
Framework for Learning Graph Distance Functions](https://arxiv.org/abs/2112.13143)
accepted at Neurips 2022.

- `data` contains `.pt` files for data used for training and evaluation

- `runlogs` contains training runlogs along with `.pt` files for pre-trained models

Place the directories under `greed/expts/` where `greed/` is `git clone`'d from the
[official repo](https://github.com/idea-iitd/greed).


The generated data is stored as Python pickle files using `torch.save()` and is in the following format: (list of queries as Pytorch-Geometric Data objects, list of targets as Pytorch-Geometric Data objects, a Pytorch tensor containing lower bounds, a Pytorch tensor containing upper bounds)

We use two kinds of data:

1. __inner dataset:__ number of pairs = number of queries = number of targets, a pair consists of a query and its corresponding target and there is a lower bound and an upper bound for each pair

2. __outer dataset:__ number of pairs = number of queries * number of targets, there is pair for each query with every target and there is a lower bound and an upper bound for each pair

Please use the `.pt` files in `data/` to use the generated data in the notebooks directly. To generate new data, run the script `data/generate_data.py` with modifications to the parameters as required.

