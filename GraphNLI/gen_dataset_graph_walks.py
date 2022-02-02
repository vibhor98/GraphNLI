"""Using various Graph walks, generate Kialo train and test sets."""

import os
import csv
import pickle as pkl
import pandas as pd
import random
import math


# Biased Root-seeking Random Walk
def biased_random_walk(sentences, data, node_id, child_edges, walk_len):
    length = 0
    label = -1
    sentences[0] = data.node[node_id]['text']

    if data.node[node_id]['relation'] == 1:
        label = 1
    elif data.node[node_id]['relation'] == -1:
        label = 0

    for i in range(1, walk_len+1):
        length += 1
        choices = []
        probs = []
        edge = data.edge[node_id]
        if len(edge.keys()) > 0:
            choices.append(list(edge.keys())[0])
            probs.append(0.75)
        if node_id in child_edges:
            choices.extend(child_edges[node_id])
            num_child = len(child_edges[node_id])
            probs.extend([0.25/num_child]*num_child)

        if len(choices) == 0:
            return sentences, label
        node = random.choices(choices, probs)[0]
        sentences[i] = data.node[node]['text']
        node_id = node
    return sentences, label


# Weighted Root-seeking Graph Walk
def weighted_graph_walk(sentences, data, node_id, walk_len):
    sentences[0] = data.node[node_id]['text']
    edge = data.edge[node_id]
    label = -1

    for i in range(1, walk_len+1):
        if len(edge.keys()) >= 1:
            parent_node_id = list(edge.keys())[0]
            sentences[i] = data.node[parent_node_id]['text']
            if i == 1:
                if edge[parent_node_id]['weight'] == 1:
                    label = 1
                elif edge[parent_node_id]['weight'] == -1:
                    label = 0
            edge = data.edge[parent_node_id]
        else:
            break
    return sentences, label


# Split dataset into train and test set.
dataset_path = '../serializedGraphs/'
files = os.listdir(dataset_path)
dataset_samples = []
labels = []

for file in files:
    data = pkl.load(open(dataset_path + file, 'rb'))

    # Required for Random Walk.
    child_edges = {}
    for node_id in data.node.keys():
        edge = data.edge[node_id]
        if len(edge.keys()) > 0:
            key = list(edge.keys())[0]
            if key in child_edges:
                child_edges[key].append(node_id)
            else:
                child_edges[key] = [node_id]

    for node_id in data.node.keys():
        sentences = ['']*4

        # Required for biased root-seeking Random Walk.
        sentences, label = biased_random_walk(sentences, data, node_id, child_edges, 3)

        # Required for weighted root-seeking Graph Walk.
        sentences, label = weighted_graph_walk(sentences, data, node_id, 4)

        if label != -1:
            sentences.append(label)
            dataset_samples.append(sentences)


print('#samples:', len(dataset_samples))
random.shuffle(dataset_samples)

train_samples = dataset_samples[ : math.ceil(0.8*len(dataset_samples))]
dev_samples = dataset_samples[math.ceil(0.8*len(dataset_samples)) : ]
pd.DataFrame(train_samples, columns=['sent1', 'sent2', 'sent3', 'sent4', 'label']).to_csv('../train_graph_random_walk.csv', index=False)
pd.DataFrame(dev_samples, columns=['sent1', 'sent2', 'sent3', 'sent4', 'label']).to_csv('../test_graph_random_walk.csv', index=False)

print('#train samples:', len(train_samples))
print('#dev samples:', len(dev_samples))
