# accurate-head-pose

We release the code of the [Hybrid Coarse-fine Classification for Head Pose Estimation](https://arxiv.org/abs/1901.06778), built on top of the [deep-head-pose](https://github.com/natanielruiz/deep-head-pose).

### Pretrained model
We provide pretrained model to reproduce the same result shown in the paper.  

[AFLW2000](https://pan.baidu.com/s/1y9q0JmnA-QxaORyn5fhPKQ), password: drmz  

[AFLW](https://pan.baidu.com/s/1rj2xLINrabaqiIzvSKlGEg), password: yym5  

[BIWI](https://pan.baidu.com/s/1bZXMdGiycX4T4u0VVofQXQ), password: 8qpc  

For those who cannot have access to BaiduDisk, you can download pretrained models on [Google Drive](https://drive.google.com/drive/folders/1is55mbFHsAVbeStkIZf9LV4HdiSjJHd9?usp=sharing)

### Testing
Training and testing lists can be found in /tools, you need download corresonding dataset and update the path.  
[AFLW2000 dataset](https://pan.baidu.com/s/1GMyAC0I_x79zXmXIegpaQg), password: xr6e  

```bash
python test_hopenet.py --gpu 0 --data_dir directory-path-for-dataset --filename_list filename-list --snapshot model --dataset dataset-name 
```


### TODO
Instructions for scripts  
Better and better models  
Videos and example demo  

### Cite this work

Haofan Wang, Zhenhua Chen and Yi Zhou "Hybrid coarse-fine classification for head pose estimation." arXiv:1901.06778, 2019. ([Download](https://arxiv.org/abs/1901.06778))

Biblatex entry:

            @article{wang2019hybrid,
              title={Hybrid coarse-fine classification for head pose estimation},
              author={Wang, Haofan and Chen, Zhenghua and Zhou, Yi},
              journal={arXiv preprint arXiv:1901.06778},
              year={2019}
            }
            

### Acknowledgement
Our hybrid classification network is plug-and-play on top of the [deep-head-pose](https://github.com/natanielruiz/deep-head-pose), but it could be extended to other classification tasks easily. We thank Nataniel Ruiz for releasing deep-head-pose-Pytorch codebase. 
