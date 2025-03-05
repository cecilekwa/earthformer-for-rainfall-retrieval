# Earthformer for multi-source-to-precipitation nowcasting for INCA domain (EF4INCA)

## Introduction

Welcome to the EF4INCA repository, which is accompanying our work [Integrated nowcasting of convective precipitation with Transformer-based models using multi-source data](https://arxiv.org/abs/2409.10367)

EF4INCA is a precipitation nowcasting model that takes data from multiple sources (e.g., satellite-, groun-based observations, modelled data). It is a modified version of the [EF-Sat2Rad](https://github.com/caglarkucuk/earthformer-satellite-to-radar/) model, which heavily borrows from the original [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer) package. 

## Installation and Setup

Installation and setting up the data involves a couple of steps:

### 0) Requirements and Versions
- CUDA: To use GPU. CUDA 11.8 was available in the machines we ran the experiments with. 

### 1) Clone the repository and jump into the main directory
```bash
cd
git clone https://github.com/caglarkucuk/earthformer-multisource-to-inca
cd earthformer-multisource-to-inca
```

### 2) Create the environment and start using it via:
Once you're in the main directory:
```bash
conda create --name ef4inca_2024 --file ef4inca/Preprocess/ef4inca_carto.txt
conda activate ef4inca_2024
```

### 3) Download and preprocess the data:
It is possible to download the full dataset for further model development or use a sampled dataset to reproduce the predictions provided with the manuscript.



- The complete dataset (130GB) is available in [https://doi.org/10.5281/zenodo.13740314](https://doi.org/10.5281/zenodo.13740314). Once downloaded, unzip the dataset to the corresponding directories in `data` with no parent directory passed from the archive, e.g.:
```bash
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/Aux.tar.gz --strip-components=2 -C data/Aux/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/val.tar.gz --strip-components=2 -C data/val/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/test.tar.gz --strip-components=2 -C data/test/

tar -xvf /PATH/TO/ZENODO/DOWNLOAD/train_2019.tar.gz --strip-components=3 -C data/train/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/train_2020.tar.gz --strip-components=3 -C data/train/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/train_2021.tar.gz --strip-components=3 -C data/train/
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/train_2022.tar.gz --strip-components=3 -C data/train/
``` 

- It is possible to use sample test dataset (370MB) provided in [https://doi.org/10.5281/zenodo.13768228](https://doi.org/10.5281/zenodo.13768228), just unzip contents of the archive to `data/test`.

```bash
tar -xvf /PATH/TO/ZENODO/DOWNLOAD/test_sampled.tar.gz --strip-components=2 -C data/test/
```

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


## Running the model
In order to run the trained model for inference, download the pretrained weights provided in [https://doi.org/10.5281/zenodo.13768228](https://doi.org/10.5281/zenodo.13768228) and unzip the file `ef_inca_multisource2precip.pt` into `trained_ckpt`. 

Afterwards run the chunk below and it'll make prediction on the test samples available in `data/test`:
```bash
cd ef4inca
python train_cuboid_inca_invLinear_v24.py --pretrained
```
It is possible to use the repository on machinces without a GPU. While it's not feasible to train the model without a GPU, it's actually okay to use CPUs for inference. In case of CPU usage, it's advisable to use lightweight, optimized approximators like [ONNX](https://onnx.ai/).

In order to train the model from scratch, run:
```bash
cd ef4inca
python train_cuboid_inca_invLinear_v24.py
```
to train the model with default parameters and original structure described in the manuscript. Modifying the model structure by creating new config files is the best way to experiment further.

## Credits
This repository is built on top of the repositories: [EF-Sat2Rad](https://github.com/caglarkucuk/earthformer-satellite-to-radar/) and [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer)


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
