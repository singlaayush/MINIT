# MINIT
This repository houses the official implementation of [Multiple Instance NeuroImage Transformer (MINiT) paper](https://arxiv.org/abs/2208.09567), accepted at PRedictive Intelligence in MEdicine (PRIME 2022) in Singapore to be held on September 24, 2022.

Here, we share the PyTorch implementation for the base NiT and the MINiT model from the paper. 
We will share the model checkpoint weights soon.

We additionally share the DistributedDataParallel-based training pipeline used to train the models as well. Please note that due to dataset access limitations with the ABCD and NCANDA datasets, we are unable to share the preprocessed dataset and labels publicly. To easily use our provided training loop, we would recommend implementing your own `get_dataset` function in the `dataset.py` for the dataset you'd like to use. We have included our own `get_dataset` code as a reference.

The training pipeline can be executed through `train.py` in this manner:
```
python train.py <model_name> <data_directory> <checkpoint_directory> <checkpoint_to_load_filename>
```
Only `model_name` is required, the rest of the arguments are optional.