# GraphNLI: A Graph-based Natural Language Inference Model for Polarity Prediction in Online Debates
**Vibhor Agarwal**, Sagar Joglekar, Anthony P. Young and Nishanth Sastry, "GraphNLI: A Graph-based Natural Language Inference Model for Polarity Prediction in Online Debates", The ACM Web Conference (TheWebConf), 2022.

## Abstract
Online forums that allow participatory engagement between users have been transformative for public discussion of important issues. However, debates on such forums can sometimes escalate into full blown exchanges of hate or misinformation. An important tool in understanding and tackling such problems is to be able to infer the argumentative relation of whether a reply is supporting or attacking the post it is replying to. This so called polarity prediction task is difficult because replies may be based on external context beyond a post and the reply whose polarity is being predicted. We propose GraphNLI, a novel graph-based deep learning architecture that uses graph walk techniques to capture the wider context of a discussion thread in a principled fashion. Specifically, we propose methods to perform root-seeking graph walks that start from a post and captures its surrounding context to generate additional embeddings for the post. We then use these embeddings to predict the polarity relation between a reply and the post it is replying to. We evaluate the performance of our models on a curated debate dataset from Kialo, an online debating platform. Our model outperforms relevant baselines, including S-BERT, with an overall accuracy of 83%.

The paper PDF will be available soon!

## Overview
**GraphNLI** is a graph-based deep learning architecture for polarity prediction, which captures both the local and the global context of the online debates through graph walks.

<div align="center">
  <img src="https://github.com/vibhor98/GraphNLI/blob/main/images/GraphNLIArch.png">
</div>

## Directory Structure
* `GraphNLI` folder contains the implementation of Graph Walks and GraphNLI model.
* `Baselines` folder contains the implementation of all the four baselines in the paper.

## Kialo Dataset
Please email us to get the Kialo dataset of online debates.

An example of the arguments made in a Kialo debate:

<div align="center">
  <img src="https://github.com/vibhor98/GraphNLI/blob/main/images/pro_life_pro_choice_arguments.png">
</div>

## Citation
If you find this paper useful in your research, please consider citing:
```
@inproceedings{agarwal2022graphnli,
  title={GraphNLI: A Graph-based Natural Language Inference Model for Polarity Prediction in Online Debates},
  author={Vibhor Agarwal and Sagar Joglekar and Anthony P. Young and Nishanth Sastry},
  booktitle={The ACM Web Conference (TheWebConf)},
  year={2022}
}
```
