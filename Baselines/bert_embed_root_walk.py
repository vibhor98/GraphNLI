"""Feature aggregation based on root-seeking Graph Walks in discussion trees with
ML and DL classifiers (without end-to-end)."""

import argparse, time
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import random


class NNet(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 n_hidden,
                 n_layers):
        super(NNet, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, features):
        h = features
        h = self.dropout(h)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.tanh(h)
                h = self.dropout(h)
            h = layer(h)
        h = self.sigmoid(h)
        return h


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, embeddings, labels):
        'Initialization'
        self.labels = labels
        self.embeddings = embeddings

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        X = self.embeddings[index]
        y = self.labels[index]
        return X, y


def weighted_root_walk(edge_dict, feat_matrix):
    queue = [(0, 0)]
    new_feat_matrix = torch.zeros((len(feat_matrix), 768*2), dtype=torch.float32)
    while len(queue) != 0:
        node_id, parent_node_id = queue.pop(0)
        if node_id in edge_dict:
            queue.extend([(n, node_id) for n in edge_dict[node_id]])

        new_feat_matrix[node_id] = torch.cat((
            feat_matrix[node_id], feat_matrix[parent_node_id]), -1)
        feat_matrix[node_id] = torch.sum(torch.stack([
            torch.mul(feat_matrix[node_id], 0.75), torch.mul(feat_matrix[parent_node_id], 0.25)]), dim=0)
    return new_feat_matrix


def node_parent_embed(edge_dict, feat_matrix):
    queue = [(0, 0)]
    new_feat_matrix = torch.zeros((len(feat_matrix), 768*2), dtype=torch.float32)
    while len(queue) != 0:
        node_id, parent_node_id = queue.pop(0)
        if node_id in edge_dict:
            queue.extend([(n, node_id) for n in edge_dict[node_id]])
        new_feat_matrix[node_id] = torch.cat((
            feat_matrix[node_id], feat_matrix[parent_node_id]), -1)
    return new_feat_matrix


def weighted_random_walk(edge_dict, node_id, node_parent_dict, walk_len):
    length = 0
    random_walk = [node_id]
    for i in range(walk_len):
        length += 1
        choices = []
        probs = []
        if node_id in node_parent_dict:
            choices.append(node_parent_dict[node_id])
            probs.append(0.75)
        if node_id in edge_dict:
            choices.extend(edge_dict[node_id])
            num_child = len(edge_dict[node_id])
            probs.extend([0.25/num_child]*num_child)
        # print(node_id, choices, probs)
        if len(choices) == 0:
            return random_walk
        node = random.choices(choices, probs)[0]
        random_walk.append(node)
        node_id = node
    return random_walk


def avg_embed(random_walk, feat_matrix):
    # try concat with random walk + new classifers like NNs, auto-encoders, etc.
    node_id = random_walk[0]
    node_embed = feat_matrix[node_id]
    node_embed_list = [node_embed]
    for node in random_walk[1:]:
        # node_embed = torch.mean(torch.stack([node_embed, feat_matrix[node]]), dim=0)
        node_embed_list.append(feat_matrix[node])

    return torch.cat(node_embed_list, -1)
    # return node_embed


def compute_agg_node_embeddings(edge_dict, feat_matrix, node_parent_dict, walk_len):
    queue = [(0, 0)]
    # new_feat_matrix = feat_matrix
    new_feat_matrix = torch.zeros((len(feat_matrix), 768*(walk_len+1)), dtype=torch.float32)
    while len(queue) != 0:
        node_id, parent_node_id = queue.pop(0)
        random_walk = weighted_random_walk(edge_dict, node_id, node_parent_dict, walk_len)

        # compute node embed following the random walk.
        # new_feat_matrix[node_id] = avg_embed(random_walk, feat_matrix)
        new_node_embed = avg_embed(random_walk, feat_matrix)
        new_feat_matrix[node_id][:len(new_node_embed)] = new_node_embed

        if node_id in edge_dict:
            queue.extend([(n, node_id) for n in edge_dict[node_id]])
    return new_feat_matrix


# data, labels = dgl.load_graphs('kialo_dgl_graphs.dgl')
data, labels = dgl.load_graphs('kialo_dgl_graphs_sbert_cls.dgl')
all_embeddings = torch.tensor(data=[])
all_labels = torch.tensor(data=[])

for i in range(len(data)):
    edge_src, edge_dest = data[i].edges()
    edge_dict = {}
    node_parent_dict = {}
    for indx in range(len(edge_src)):
        if int(edge_src[indx]) not in edge_dict:
            edge_dict[int(edge_src[indx])] = [int(edge_dest[indx])]
        else:
            edge_dict[int(edge_src[indx])].append(int(edge_dest[indx]))
        node_parent_dict[int(edge_dest[indx])] = int(edge_src[indx])
    data[i].ndata['feat'] = weighted_root_walk(edge_dict, data[i].ndata['feat'])

    # data[i].ndata['feat'] = compute_agg_node_embeddings(
    #                     edge_dict, data[i].ndata['feat'], node_parent_dict, 2)

    if i == 0:
        all_embeddings = data[i].ndata['feat']
        all_labels = data[i].ndata['label']
    else:
        all_embeddings = torch.cat((all_embeddings, data[i].ndata['feat']), dim=0)
        all_labels = torch.cat((all_labels, data[i].ndata['label']), dim=0)

    # num_nodes = len(data[i].nodes())
    # num_nodes_train = int(num_nodes * 0.8)
    # print('No. of nodes in train set:', num_nodes_train)
    # print('No. of nodes in test set:', num_nodes - num_nodes_train)
    #
    # clf = LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear', random_state=0)
    #
    # clf = clf.fit(data[i].ndata['feat'][:num_nodes_train], data[i].ndata['label'][:num_nodes_train])
    # pred = clf.predict(data[i].ndata['feat'][num_nodes_train:])
    # print(pred)
    # print(data[i].ndata['label'][num_nodes_train:])
    # print(accuracy_score(pred, data[i].ndata['label'][num_nodes_train:]))
    # print(precision_score(pred, data[i].ndata['label'][num_nodes_train:]))
    # print(recall_score(pred, data[i].ndata['label'][num_nodes_train:]))
    # print(f1_score(pred, data[i].ndata['label'][num_nodes_train:]))


############# logistic regression model
num_nodes = len(all_embeddings)
num_nodes_train = int(num_nodes * 0.8)
print('No. of nodes in train set:', num_nodes_train)
print('No. of nodes in test set:', num_nodes - num_nodes_train)

clf = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0)

clf = clf.fit(all_embeddings[:num_nodes_train], all_labels[:num_nodes_train])
pred = clf.predict(all_embeddings[num_nodes_train:])
print(Counter(pred))
print(precision_score(pred, all_labels[num_nodes_train:]))
print(recall_score(pred, all_labels[num_nodes_train:]))
print(f1_score(pred, all_labels[num_nodes_train:]))
print(accuracy_score(pred, all_labels[num_nodes_train:]))


############# Neural Net
training_set = Dataset(all_embeddings[:num_nodes_train], all_labels[:num_nodes_train])
testing_set = Dataset(all_embeddings[num_nodes_train:], all_labels[num_nodes_train:])

train_dataloader = DataLoader(
    training_set, batch_size=32, shuffle=True, drop_last=False)
test_dataloader = DataLoader(
    testing_set, batch_size=32, shuffle=True, drop_last=False)

in_feats = len(all_embeddings[0])
n_classes = 2

model = NNet(in_feats, n_classes, 768, 1)
print(model)

loss_fcn = torch.nn.CrossEntropyLoss()

# use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(500):
    t0 = time.time()
    pred_all = []
    labels_all = []
    loss_all = []
    for batch, labels in train_dataloader:
        # forward
        logits = model(batch)
        pred = logits.argmax(1)
        loss = loss_fcn(logits, labels)

        pred_all.extend(pred)
        labels_all.extend(labels)
        loss_all.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    dur = time.time() - t0

    acc = accuracy_score(pred_all, labels_all)

    pred_all = []
    labels_all = []
    for batch, labels in test_dataloader:
        logits = model(batch)
        pred = logits.argmax(1)
        pred_all.extend(pred)
        labels_all.extend(labels)

    print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | Train Accuracy {:.4f} | Test Accuracy {:.4f}"
         .format(epoch, dur, np.mean(loss_all), acc, accuracy_score(pred_all, labels_all)))

model.eval()
pred_all = []
labels_all = []
for batch, labels in test_dataloader:
    logits = model(batch)
    pred = logits.argmax(1)
    pred_all.extend(pred)
    labels_all.extend(labels)

print("Precision:", precision_score(pred_all, labels_all))
print("Recall:", recall_score(pred_all, labels_all))
print("F1-score:", f1_score(pred_all, labels_all))
print("Accuracy:", accuracy_score(pred_all, labels_all))
