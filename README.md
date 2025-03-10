# Earthformer for rainfall retrieval

## Introduction

Welcome to the Earthformer for rainfall repository, which is accompanying the master thesis work [Implementation of a space-time transformer retrieval algorithm for high rainfall rates from the Meteosat satellite applied in Africa](https://repository.tudelft.nl/record/uuid:0b37d583-e915-4a2a-9e60-55e250f62421)

the Earthformer for rainfall retrieval model  is a rainfall retrieval model that takes data from multiple open sources.It is a modified version of the [EF4INCA](https://github.com/caglarkucuk/earthformer-multisource-to-inca/tree/main/data) model, which heavily borrows from the original [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer) package. 

The data used in this research is listed below. 

**Input data**
* Meteosat Second Generation (MSG) satellite images of the Spinning Enhanced Visual Infrared Imager (SEVIRI)
* Digital Elevation model (DEM)
* Longitude, latitude grid
* Time of the Day, Day of the Year

**Reference data**
* IMERG-Final

**Benchmark data**
* IMERG-Early

## Installation and Setup

Installation and setting up the data involves a couple of steps:

### 0) Requirements and Versions
- CUDA: To use GPU. CUDA 11.8 was available in the machines we ran the experiments with. 

### 1) Clone the repository and jump into the main directory
```bash
cd
git clone https://github.com/cecilekwa/earthformer-for-rainfall-retrieval
cd earthformer-multisource-to-inca
```

### 2) Create the environment and start using it via:
Once you're in the main directory:
```bash
conda create --name ef4inca_2024 --file ef4inca/Preprocess/ef4inca_carto.txt
conda activate ef4inca_2024
```

### 3) Download and preprocess the data:
It is possible to download input dataset, used in for training the model for further model development or use the test dataset and pretrained model weights to reproduce the estimations. When it is desired to apply pretrained model to different locations and/or timesteps, 



<!-- - The complete dataset (130GB) is available in [https://doi.org/10.5281/zenodo.13740314](https://doi.org/10.5281/zenodo.13740314). Once downloaded, unzip the dataset to the corresponding directories in `data` with no parent directory passed from the archive, e.g.:
```bash
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/Aux.tar.gz --strip-components=2 -C data/Auxillary/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/val.tar.gz --strip-components=2 -C data/val/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/test.tar.gz --strip-components=2 -C data/test/

tar -xvf /PATH/TO/ZENODO/DOWNLOAD/train_2019.tar.gz --strip-components=3 -C data/train/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/train_2020.tar.gz --strip-components=3 -C data/train/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/train_2021.tar.gz --strip-components=3 -C data/train/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/train_2022.tar.gz --strip-components=3 -C data/train/ -->
``` 

- It is possible to download the pretrained model weights, auxillary data and test dataset (13.3GB) provided in [https://doi.org/10.5281/zenodo.14993091](https://doi.org/10.5281/zenodo.14993091), just unzip contents of the archive of the test dataset to `data/test`, the auxillary dataset to 'data/auxillary' and the desired pretrained model weights *ef_inca_multisource2precip.pt* of either the Earthformer model trained with MSE loss function or the balanced weighting loss function in the 'trained_ckpt' datafolder, without 

<!-- ```bash
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/test_sampled.tar.gz --strip-components=2 -C data/test/
``` -->

## Pre-processing the data

### 1) Dowloading the data
The model uses fully open source data to train the model on. 

* The model uses SEVIRI channel data as input, which can be downloaded from the EUMDAC database using the EUMDAC python library. The script ***seviri_retrieval_nat_to_h5.ipynb*** can be used for this. Note that using this script your data is also already reprojected. 

* The model is trained on IMERG-Final, which can be downloaded using the GPM-API python library with the ***imerg_retrieval.ipynb*** script. 

* Other data that is used is the Digital Elevation Model (DEM), which you can be obtained from the NASA database and the longitude and latitude grid. 

* The benchmark IMERG-Early can be downloaded using the same script as the ***imerg_retrieval.ipynb*** script. 

### 2) Resampling the data
* If the SEVIRI data was not reprojected yet, the ***seviri_retrieval_reproject_nat_to_hdf5.ipynb*** can be used to reproject the geostationary projection into WGS84 projection.
* The IMERG data can be resampled to the same resolution as the SEVIRI data using the ***imerg_resampling.ipynb*** script. 

### 3) Selecting data
The selection of data is based on the values of the IMERG data. It is done in a two-step approach.
* First, all the selection criteria are saved into a dataframe, so later, the selection criteria can be decided upon more iteratively. This is done using the ***1.retrieve_selection_criteria.ipynb*** script. 
* Secondly, the selected timestamps are used to merge the SEVIRI data and IMERG data into one file, which is currently required for using the model. This is done, using the ***2.event_selection_merging_files.ipynb***.

### 4) Splitting data
Although it is advisable to use a different splitting method compared to what was used in this research it was left here for reproducibility. 
The script to split the data into a training, test and validation dataset and save it into the correct directories using the script ***train_test_val_splitting.ipynb***.

### 5) Normalizing data
The SEVIRI data is normalized for better performance of the model. The IMERG data is log transformed in the dataUtils script of the model. To obtain the mean and standard deviation of the entire dataset of the seviri dataset the ***norm_values_input_data.ipynb*** can be used. 

## The model explained

1) Earthformer architecture
   A schematic overview of the Earthformer model, with the encoder on the left and the decoder on the right. The input tensor first passes through the 2D-CNN and Downsample layers, which reduce its spatial dimensions while increasing its channel dimensions through convolution, patch merging, tokenizing the data. The tokenized data is then processed by the cuboid attention block to capture contextual relationships. Here, M represents the number of hierarchical layers, while D denotes the number of cuboids within each layer. The model's upsampling is performed using Nearest Neighbour Interpolation.

![image](https://github.com/user-attachments/assets/5f27df0f-b318-4a78-a23f-a5b23fb07641)

2) The Figure shows how the attention mechanism within the cuboid is decomposed in smaller cuboids, for computational efficiency. Within each cuboid the attention mechanism is run, while all cuboids also attend to the global parameter to ensure system wide dynamics are captured as well. After the attention mechanism has run, the decompased blocks are merged back together.

3) The Figure shows different strategies to decompose the cuboid attention block.

## Model architecture

This table provides an overview of the model layers, showing the transformations applied at each stage, including changes in resolution and channel dimensions.

| **Block** | **Layer** | **Resolution** | **Channels** |
|-----------|----------|---------------|--------------|
| **Input** | - | 248 × 184 | 11 |
| **2D CNN + Downsampler** | Conv3 × 3  <br> Conv3 × 3 <br> GroupNorm16 <br> LeakyReLU <br> PatchMerge <br> LayerNorm <br> Linear | 248 × 184 <br> 248 × 184 <br> 248 × 184 <br> 248 × 184 <br> 248 × 184 → 83 × 62 <br> 83 × 62 <br> 83 × 62 | 11 → 32 <br> 32 <br> 32 <br> 32 <br> 32 → 288 <br> 288 <br> 288 → 32 |
| **2D CNN + Downsampler** | Conv3 × 3  <br> Conv3 × 3 <br> GroupNorm16 <br> LeakyReLU <br> PatchMerge <br> LayerNorm <br> Linear | 83 × 62 <br> 83 × 62 <br> 83 × 62 <br> 83 × 62 <br> 83 × 62 → 42 × 31 <br> 42 × 31 <br> 42 × 31 | 32 → 128 <br> 128 <br> 128 <br> 128 <br> 128 → 512 <br> 512 <br> 512 → 128 |
| **Encoder Positional Embedding** | PosEmbed | 42 × 31 | 128 |
| **Cuboid Attention Block x 1** | LayerNorm <br> Cuboid (T,1,1) <br> FFN <br> LayerNorm <br> Cuboid (1,H,1) <br> FFN <br> LayerNorm <br> Cuboid (1,1,W) <br> FFN | 42 × 31 | 128 |
| **Downsampler** | PatchMerge <br> LayerNorm <br> Linear | 42 × 31 → 21 × 16 <br> 21 × 16 <br> 21 × 16 | 128 → 512 <br> 512 <br> 512 → 256 |
| **Decoder Initial Positional Embedding** | PosEmbed | 21 × 16 | 256 |
| **Cuboid Attention Block x 1** | LayerNorm <br> Cuboid (T,1,1) <br> FFN <br> LayerNorm <br> Cuboid (1,H,1) <br> FFN <br> LayerNorm <br> Cuboid (1,1,W) <br> FFN | 21 × 16 | 256 |
| **Upsampler** | Nearest Neighbour Interpolation <br> Conv3 × 3 | 21 × 16 → 42 × 31 <br> 42 × 31 | 128 <br> 128 |
| **Cuboid Attention Block x 1** | LayerNorm <br> Cuboid (T,1,1) <br> FFN <br> LayerNorm <br> Cuboid (1,H,1) <br> FFN <br> LayerNorm <br> Cuboid (1,1,W) <br> FFN | 42 × 31 | 256 |
| **2D CNN + Upsampler** | Nearest Neighbour Interpolation <br> Conv3 × 3 <br> GroupNorm16 <br> LeakyReLU | 21 × 16 → 42 × 31 <br> 42 × 31 <br> 42 × 31 <br> 42 × 31 | 256 → 64 <br> 64 <br> 64 <br> 64 |


## Running the model
In order to run the trained model for inference, download the pretrained weights provided in [https://doi.org/10.5281/zenodo.13768228](https://doi.org/10.5281/zenodo.13768228) and unzip the file `ef_inca_multisource2precip.pt` into `trained_ckpt`. 

Afterwards run the chunk below and it'll make prediction on the test samples available in `data/test`:
```bash
cd ef4inca
python train_cuboid_inca_invLinear_v24.py --pretrained
```
It is possible to use the repository on machinces without a GPU. While it's not feasible to train the model without a GPU, it's actually okay to use CPUs for inference. In case of CPU usage, it's advisable to use lightweight, optimized approximators like [ONNX](https://onnx.ai/).

## Train model from scratch

In order to train the model from scratch, run:
```bash
cd ef4inca
python train_cuboid_inca_invLinear_v24.py
```
to train the model with default parameters and original structure described in the manuscript. Modifying the model structure by creating new config files is the best way to experiment further.

In the ***train_cuboid_inca_invLinear_v24.py*** defines what happens during the training of your model. PyTorch Lightning is used for efficient training. PyTorch Lightning sets requirements to how you name certain functions and is linked to the process and or phase your model is in. 

For example; The *on_validation_epoch_end* function defines what happens at the end of a validation epoch and *training_step* function defines what happens every step of the training.

In the utils directory
The ***dataUtils_flex.py*** file prepares your data so it can be used for training, testing and validation. Data is normalized, before it is used as input for the model and it is reformatted it correct tensors. 

The ***VisUtils_elegant.py*** file defines how your data is plotted during training, validation and testing. 

In the ***FixedValues.py*** you can set the mean and standard deviation of your data, which is used in the ***dataUtils_flex.py***. 


## Credits
This repository is built on top of the repositories: [EF4INCA](https://github.com/caglarkucuk/earthformer-multisource-to-inca/tree/main/data)) and [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer).


## Cite
Please cite us if this repo helps your work!
```
@article{Kucuk2024,
   title = {Integrated nowcasting of convective precipitation with Transformer-based models using multi-source data},
   author = {K\"u\c{c}\"uk, {\c{C}}a\u{g}lar and Atencia, Aitor and Dabernig, Markus},
   doi = {10.48550/arXiv.2409.10367},
   year = {2024}
}
``` 

## Licence
GNU General Public License v3.0
