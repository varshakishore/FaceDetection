# Random Neural Network Steganography

This repository is the official implementation of [Random Neural Network Steganography](https://github.com/varshakishore/rnns). 

<p align="center"><img width="80%" src="mainfigure.png" /></p>

## Requirements

To install requirements:
1. Install [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) and [cuDNN 7.4.2](https://developer.nvidia.com/rdp/cudnn-archive).
2. Install conda environment:
    ```sh
    conda env create -f environment.yml
    python -m ipykernel install --user --name=rnns
    ```
3. Prepare images (we use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as an example):
    ```sh
    wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
    unzip -q DIV2K_valid_HR.zip
    ```

## Running Demo

To train the model(s) in the paper, run this command:

```sh
python demo.py --num_bits 3
```

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |
