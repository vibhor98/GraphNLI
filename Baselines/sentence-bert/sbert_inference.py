from torch.utils.data import DataLoader
from torch import nn
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from sentence_transformers.readers import InputExample
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModel
import pickle as pkl
import pandas as pd
import numpy as np
import csv


test_samples = []
val_label = []

# Our S-BERT trained model path.
model_path = 'sbert_nli_model'

devset = pd.read_csv('./test_graph_set.csv')
for i in range(len(devset)):
    test_samples.append([str(devset.iloc[i]['sentence1']), str(devset.iloc[i]['sentence2'])])
    val_label.append(int(devset.iloc[i]['label']))

model = CrossEncoder(model_path, num_labels=2)

pred_prob = model.predict(test_samples, activation_fct=nn.Softmax())

pred_labels = np.argmax(pred_prob, axis=1)

print('Precision:', precision_score(val_label, pred_labels))
print('Recall:', recall_score(val_label, pred_labels))
print('F1-score:', f1_score(val_label, pred_labels))

print('Classification Report')
print(classification_report(val_label, pred_labels))
