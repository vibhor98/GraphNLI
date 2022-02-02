"""Implementation of GraphNLI model."""

import math
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os
import gzip
import csv
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from torch.utils.data import DataLoader
from SoftmaxLoss import *


model_name = sys.argv[1] if len(sys.argv) > 1 else 'distilroberta-base'

train_batch_size = 16
graph_walk_len = 5
random_walk_len = 4
num_epochs = 4

train_samples = []
test_samples = []

model_save_path = 'output/training_nli_' + model_name.replace("/", "-")

# Using RoBERTa model for mapping tokens to embeddings.
word_embedding_model = models.Transformer(model_name)

# Applying mean pooling to get one fixed sized sentence vector.
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# trainset = pd.read_csv('../train_graph_set_walk.csv')
trainset = pd.read_csv('../train_graph_random_walk.csv')
trainset = trainset.fillna('')

for i in range(len(trainset)):
    texts = []
    for j in range(1, random_walk_len+1):  # 6 for graph walk and 5 for random walk.
            texts.append(trainset.iloc[i]['sent' + str(j)])
    train_samples.append(InputExample(texts=texts, label=int(trainset.iloc[i]['label'])))

devset = pd.read_csv('../test_graph_random_walk.csv')
devset = devset.fillna('')

for i in range(len(devset)):
    texts = []
    for j in range(1, random_walk_len+1):
        texts.append(devset.iloc[i]['sent' + str(j)])
    test_samples.append(InputExample(texts=texts, label=int(devset.iloc[i]['label'])))

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)

dev_dataloader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)

dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, name='sts-dev', softmax_model=train_loss)


warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )


# Load the stored model and evaluate its performance on the test set.
test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)

model = SentenceTransformer(model_save_path)
test_loss = SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)
test_evaluator = LabelAccuracyEvaluator(test_dataloader, name='sts-test', softmax_model=test_loss)
test_evaluator(model, output_path=model_save_path)
