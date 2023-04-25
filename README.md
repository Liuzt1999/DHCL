# Deep Hash Remote-Sensing Image Retrieval Assisted by Semantic Cues

Offical Pytorch implementation of Remote Sensing papar [Deep Hash Remote-Sensing Image Retrieval Assisted by Semantic Cues](https://doi.org/10.3390/rs14246358)

![image](https://github.com/Liuzt1999/DHCL/blob/main/material/GA.png)

An network architecture, which can classify and retrieve remote-sensing images under a unified framework, and the classification labels are further utilized as the semantic cues to assist in network training.

A standard hash code structure which integrate the classification results into the hash-retrieval process to improve accuracy.

## Requirment

- Python3
- Pytorch(>1.0)
- Numpy
- tqdm
- wandb
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

## Datasets
Download public benchmarks
- UCMD([Link](http://weegee.vision.ucmerced.edu/datasets/landuse.html))
- AID([Link](https://pan.baidu.com/s/1mifOBv6#list/path=%2F))

## Result

Note that a sufficiently large batch size and good parameters resulted in better overall performance than that described in the paper.
Larger hash code size means better results with greater consumption.

### UCMD
The Evaluation Metrics is MAP@20
| Method | Backbone | 16bits | 32bits | 48bits | 64bits |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Ours | Inception-BN | 98.97 | 99.34 | 99.54 | 99.60 |

### AID
The Evaluation Metrics is MAP@20
| Method | Backbone | 16bits | 32bits | 48bits | 64bits |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Ours | Inception-BN | 94.75 | 98.08 | 98.93 | 99.02 |

## Acknowledgements

Our code is modified and adapted on these great repositories:

- [PyTorch Metric learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- [[2003.13911\] Proxy Anchor Loss for Deep Metric Learning (arxiv.org)](https://arxiv.org/abs/2003.13911)

## Citation

If you use this method or this code in your research, please cite as:

    @article{
    title = {Deep Hash Remote-Sensing Image Retrieval Assisted by Semantic Cues},
    journal = {Remote Sensing},
    volume = {14},
    pages = {2358},
    year = {2022},
    doi = {https://doi.org/10.3390/rs14246358},
    url = {https://www.mdpi.com/2072-4292/14/24/6358#},
    author = {Pingping Liu and Zetong Liu and Xue Shan and Qiuzhan Zhou},
    keywords = {Remote sensing, Image retrieval, Deep hash, Metric learning},
    abstract = {With the significant and rapid growth in the number of remote-sensing images, deep hash methods have become a research topic. The main work of deep hash method is to build a discriminate embedding space through the similarity relation between sample pairs and then map the feature vector into Hamming space for hashing retrieval. We demonstrate that adding a binary classification label as a kind of semantic cue could further improve the retrieval performance. In this work, we propose a new method, which we called deep hashing, based on classification label (DHCL). First, we propose a network architecture, which can classify and retrieve remote-sensing images under a unified framework, and the classification labels are further utilized as the semantic cues to assist in network training. Second, we propose a hash code structure, which can integrate the classification results into the hash-retrieval process to improve accuracy. Finally, we validate the performance of the proposed method on several remote-sensing image datasets and show the superiority of our method.}
    }
