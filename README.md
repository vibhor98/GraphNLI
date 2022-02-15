# GraphNLI: A Graph-based Natural Language Inference Model for Polarity Prediction in Online Debates
**Vibhor Agarwal**, Sagar Joglekar, Anthony P. Young and Nishanth Sastry, "GraphNLI: A Graph-based Natural Language Inference Model for Polarity Prediction in Online Debates", The ACM Web Conference (TheWebConf), 2022.

## Abstract
An online forum that allows participatory engagement between users, very often, becomes a stage for heated debates. These debates sometimes escalate into full blown exchanges of hate and misinformation. As such, modeling these conversations through the lens of argumentation theory as graphs of supports and attacks has shown promise, especially in identifying which claims should be accepted. However, the argumentative relation of supports and attacks, also called the *polarity*, is difficult to infer from natural language exchanges, not least because support or attack relationship in natural language is intuitively contextual.

Various deep learning models have been used to classify the polarity, where the inputs to the model are typically just the texts of the replying argument and the argument being replied to. We propose GraphNLI, a novel graph-based deep learning architecture to infer argumentative relations, which not only considers the immediate pair of arguments involved in the response, but also the surrounding arguments, hence capturing the context of the discussion, through graph walks. We demonstrate the performance of this model on a curated debate dataset from Kialo, an online debating platform. Our model outperforms the relevant baselines with an overall accuracy of 83%, which demonstrates that incorporating nearby arguments in addition to the pair of relayed arguments helps in predicting argumentative relations in online debates.

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
