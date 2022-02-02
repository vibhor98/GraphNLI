from torch.utils.data import DataLoader
from torch import nn
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from sentence_transformers.readers import InputExample
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModel
import os
import random
import pickle as pkl
import pandas as pd
import numpy as np
import csv

# Split dataset into train and test set.

dataset_path = './serializedGraphs/'
files = os.listdir(dataset_path)
dataset_samples = []
labels = []

for file in files:
    data = pkl.load(open(dataset_path + file, 'rb'))
    for node_id in data.node.keys():
        sentence2 = data.node[node_id]['text']
        edge = data.edge[node_id]

        if len(edge.keys()) >= 1:
            parent_node_id = list(edge.keys())[0]
            if edge[parent_node_id]['weight'] == 1:
                sentence1 = data.node[parent_node_id]['text']
                dataset_samples.append([sentence1, sentence2, 1])
            elif edge[parent_node_id]['weight'] == -1:
                sentence1 = data.node[parent_node_id]['text']
                dataset_samples.append([sentence1, sentence2, 0])

print('#samples:', len(dataset_samples))
random.shuffle(dataset_samples)

train_samples = dataset_samples[ : math.ceil(0.8*len(dataset_samples))]
dev_samples = dataset_samples[math.ceil(0.8*len(dataset_samples)) : ]
pd.DataFrame(train_samples, columns=['sentence1', 'sentence2', 'label']).to_csv('train_graph_set.csv', index=False)
pd.DataFrame(dev_samples, columns=['sentence1', 'sentence2', 'label']).to_csv('test_graph_set.csv', index=False)

print('#train samples:', len(train_samples))
print('#dev samples:', len(dev_samples))

##########################################
train_samples = []
dev_samples = []
test_samples = []
val_label = []

trainset = pd.read_csv('./train_graph_set.csv')
for i in range(len(trainset)):
    train_samples.append(InputExample(texts=[str(trainset.iloc[i]['sentence1']),
            str(trainset.iloc[i]['sentence2'])], label=int(trainset.iloc[i]['label'])))

devset = pd.read_csv('./test_graph_set.csv')
for i in range(len(devset)):
    dev_samples.append(InputExample(texts=[str(devset.iloc[i]['sentence1']),
            str(devset.iloc[i]['sentence2'])], label=int(devset.iloc[i]['label'])))
    test_samples.append([str(devset.iloc[i]['sentence1']), str(devset.iloc[i]['sentence2'])])
    val_label.append(int(devset.iloc[i]['label']))

train_batch_size = 32
num_epochs = 4
model_save_path = 'sbert_nli_model'

#Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 2 labels
model = CrossEncoder('distilroberta-base', num_labels=2)

#We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

#During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CEBinaryAccuracyEvaluator.from_input_examples(dev_samples, name='nli-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          activation_fct=nn.Softmax(),
          evaluation_steps=10000,
          optimizer_params={'lr': 1e-5},
          warmup_steps=warmup_steps,
          output_path=model_save_path)
