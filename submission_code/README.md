# Random Neural Network Steganography

This repository is the official implementation of [Random Neural Network Steganography](https://github.com/varshakishore/rnns). 

<p align="center"><img width="80%" src="mainfigure.png" /></p>

## Requirements

To install requirements:
1. Install [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) and [cuDNN 7.4.2](https://developer.nvidia.com/rdp/cudnn-archive).
2. Install conda environment:
    ```sh
    conda create -n rnns python=3.6
    conda activate rnns
    conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
    conda install ipykernel imageio tqdm
    python -m pip install reedsolo
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

The following results are also reported in our paper.

Comparison of various flavors of RNNS and SOTA image steganography algorithms on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset.
<table>
    <thead>
        <tr>
            <td colspan=2, align='center'>Method</td>
            <th>1 BPP</th>
            <th>2 BPP</th>
            <th>3 BPP</th>
            <th>4 BPP</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=2, align='center'><a href="https://papers.nips.cc/paper/2017/file/838e8afb1ca34354ac209f53d90c3a43-Paper.pdf">Deep Steganography</a></td>
            <td align='center'>1.7%±0.6</td>
            <td align='center'>3.1%±0.6</td>
            <td align='center'>5.2%±0.6</td>
            <td align='center'>10.4%±1.1</td>
        </tr>
        <tr>
            <td colspan=2, align='center'><a href="https://arxiv.org/pdf/1901.03892.pdf">SteganoGAN</a></td>
            <td align='center'>1.7%±1.0</td>
            <td align='center'>3.0%±0.9</td>
            <td align='center'>5.1%±1.3</td>
            <td align='center'>9.3%±1.8</td>
        </tr>
        <tr>
            <td colspan=2, align='center'>RNNS</td>
            <td align='center'>2e-2%±0.1</td>
            <td align='center'>0.2%±0.3</td>
            <td align='center'>3.3%±2.6</td>
            <td align='center'>10.9%±4.8</td>
        </tr>
        <tr>
            <td colspan=2, align='center'>RNNS (Pre-trained decoder)</td>
            <td align='center'>0.4%±0.9</td>
            <td align='center'>0.4%±0.9</td>
            <td align='center'>14.8%±4.6</td>
            <td align='center'>18.6%±3.9</td>
        </tr>
        <tr>
            <td rowspan=5, align='center'>RNNS<br>+<br>Hamming</td>
            <td align='center'>Restart 0x</td>
            <td align='center'>0%±0</td>
            <td align='center'>0.5%±1.2</td>
            <td align='center'>6.7%±6.5</td>
            <td align='center'>16.1%±9.2</td>
        </tr>
        <tr>
            <td align='center'>Restart 1x</td>
            <td align='center'>0%±0</td>
            <td align='center'>0.2%±0.5</td>
            <td align='center'>3.6%±3.8</td>
            <td align='center'>11.1%±7.5</td>
        </tr>
        <tr>
            <td align='center'>Restart 2x</td>
            <td align='center'>0%±0</td>
            <td align='center'>2e-3%±4e-3</td>
            <td align='center'>6.0%±1.6</td>
            <td align='center'>6.0%±6.0</td>
        </tr>
        <tr>
            <td align='center'>Restart 3x</td>
            <td align='center'>0%±0</td>
            <td align='center'>9e-4%±3e-3</td>
            <td align='center'>0.7%±1.0</td>
            <td align='center'>4.3%±4.0</td>
        </tr>
        <tr>
            <td align='center'>Restart 4x</td>
            <td align='center'><b>0%±0</b></td>
            <td align='center'><b>5e-4%±2e-3</b></td>
            <td align='center'><b>0.5%±1.0</b></td>
            <td align='center'><b>4.1%±3.8</b></td>
        </tr>
    </tbody>
</table>

Performance of RNNS on different datasets with varying values of ε.

<table>
    <thead>
        <tr>
            <th></th>
            <th>BPP</th>
            <td colspan=3, align='center'>ε=0.1</td>
            <td colspan=3, align='center'>ε=0.2</td>
            <td colspan=3, align='center'>ε=0.3</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4, align='center'><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">div2k</a></td>
            <td align='center'>1</td>
            <td align='center'>0.02 (0.04)</td>
            <td align='center'>32.88 (4.72)</td>
            <td align='center'>0.85 (0.15)</td>
            <td align='center'>0.0 (0.01)</td>
            <td align='center'>31.3 (7.21)</td>
            <td align='center'>0.78 (0.26)</td>
            <td align='center'>0.02 (0.11)</td>
            <td align='center'>35.31 (3.48)</td>
            <td align='center'>0.93 (0.09)</td>
        </tr>
        <tr>
            <td align='center'>2</td>
            <td align='center'>0.9 (0.82)</td>
            <td align='center'>30.03 (2.69)</td>
            <td align='center'>0.8 (0.12)</td>
            <td align='center'>0.14 (0.29)</td>
            <td align='center'>27.49 (5.13)</td>
            <td align='center'>0.69 (0.24)</td>
            <td align='center'>0.18 (0.31)</td>
            <td align='center'>30.73 (3.6)</td>
            <td align='center'>0.85 (0.14)</td>
        </tr>
        <tr>
            <td align='center'>3</td>
            <td align='center'>6.09 (3.61)</td>
            <td align='center'>29.12 (2.42)</td>
            <td align='center'>0.76 (0.11)</td>
            <td align='center'>1.89 (2.08)</td>
            <td align='center'>26.78 (4.62)</td>
            <td align='center'>0.67 (0.22)</td>
            <td align='center'>3.29 (2.56)</td>
            <td align='center'>28.99 (3.95)</td>
            <td align='center'>0.78 (0.18)</td>
        </tr>
        <tr>
            <td align='center'>4</td>
            <td align='center'>15.7 (5.21)</td>
            <td align='center'>28.83 (2.27)</td>
            <td align='center'>0.74 (0.11)</td>
            <td align='center'>10.05 (5.2)</td>
            <td align='center'>26.74 (4.15)</td>
            <td align='center'>0.67 (0.2)</td>
            <td align='center'>10.88 (4.82)</td>
            <td align='center'>28.6 (3.66)</td>
            <td align='center'>0.76 (0.15)</td>
        </tr>
        <tr>
            <td rowspan=4, align='center'><a href="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">celeba</a></td>
            <td align='center'>1</td>
            <td align='center'>0.07 (0.16)</td>
            <td align='center'>37.29 (5.74)</td>
            <td align='center'>0.86 (0.2)</td>
            <td align='center'>0.05 (0.15)</td>
            <td align='center'>36.56 (7.64)</td>
            <td align='center'>0.83 (0.28)</td>
            <td align='center'>0.14 (0.35)</td>
            <td align='center'>39.79 (2.5)</td>
            <td align='center'>0.96 (0.04)</td>
        </tr>
        <tr>
            <td align='center'>2</td>
            <td align='center'>0.91 (0.98)</td>
            <td align='center'>32.76 (4.04)</td>
            <td align='center'>0.77 (0.18)</td>
            <td align='center'>0.64 (1.08)</td>
            <td align='center'>31.41 (6.4)</td>
            <td align='center'>0.71 (0.29)</td>
            <td align='center'>1.08 (1.46)</td>
            <td align='center'>35.12 (2.4)</td>
            <td align='center'>0.89 (0.07)</td>
        </tr>
        <tr>
            <td align='center'>3</td>
            <td align='center'>4.93 (3.88)</td>
            <td align='center'>31.29 (3.59)</td>
            <td align='center'>0.71 (0.16)</td>
            <td align='center'>3.64 (4.16)</td>
            <td align='center'>29.16 (6.1)</td>
            <td align='center'>0.62 (0.29)</td>
            <td align='center'>5.28 (3.87)</td>
            <td align='center'>33.4 (3.17)</td>
            <td align='center'>0.84 (0.12)</td>
        </tr>
        <tr>
            <td align='center'>4</td>
            <td align='center'>17.27 (6.89)</td>
            <td align='center'>31.42 (3.79)</td>
            <td align='center'>0.71 (0.16)</td>
            <td align='center'>12.42 (6.98)</td>
            <td align='center'>29.6 (5.91)</td>
            <td align='center'>0.64 (0.27)</td>
            <td align='center'>15.17 (5.83)</td>
            <td align='center'>32.53 (4.17)</td>
            <td align='center'>0.79 (0.17)</td>
        </tr>
        <tr>
            <td rowspan=4, align='center'><a href="https://cocodataset.org/#home">mscoco</a></td>
            <td align='center'>1</td>
            <td align='center'>0.07 (0.1)</td>
            <td align='center'>32.93 (4.73)</td>
            <td align='center'>0.84 (0.16)</td>
            <td align='center'>0.01 (0.05)</td>
            <td align='center'>31.23 (7.12)</td>
            <td align='center'>0.76 (0.28))</td>
            <td align='center'>0.04 (0.26)</td>
            <td align='center'>34.68 (3.85)</td>
            <td align='center'>0.91 (0.12)</td>
        </tr>
        <tr>
            <td align='center'>2</td>
            <td align='center'>1.59 (1.45)</td>
            <td align='center'>29.81 (2.51)</td>
            <td align='center'>0.76 (0.13)</td>
            <td align='center'>0.6 (2.86)</td>
            <td align='center'>26.89 (5.05)</td>
            <td align='center'>0.65 (0.25)</td>
            <td align='center'>0.32 (1.39)</td>
            <td align='center'>30.79 (3.01)</td>
            <td align='center'>0.84 (0.1)</td>
        </tr>
        <tr>
            <td align='center'>3</td>
            <td align='center'>6.46 (4.14)</td>
            <td align='center'>29.11 (2.06)</td>
            <td align='center'>0.73 (0.11)</td>
            <td align='center'>1.54 (1.34)</td>
            <td align='center'>26.15 (4.21)</td>
            <td align='center'>0.62 (0.22)</td>
            <td align='center'>2.16 (1.76)</td>
            <td align='center'>29.32 (2.9)</td>
            <td align='center'>0.79 (0.1)</td>
        </tr>
        <tr>
            <td align='center'>4</td>
            <td align='center'>15.27 (4.81)</td>
            <td align='center'>28.76 (1.83)</td>
            <td align='center'>0.71 (0.11)</td>
            <td align='center'>8.56 (3.99)</td>
            <td align='center'>25.96 (3.96)</td>
            <td align='center'>0.6 (0.22)</td>
            <td align='center'>10.38 (5.02)</td>
            <td align='center'>28.22 (3.2)</td>
            <td align='center'>0.74 (0.14)</td>
        </tr>
    </tbody>
</table>
