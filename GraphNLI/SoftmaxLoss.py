import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from sentence_transformers import SentenceTransformer
import logging


logger = logging.getLogger(__name__)

class SoftmaxLoss(nn.Module):
    """
    This loss function is a modification of loss used in S-BERT to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.
    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        logger.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = self.mean_aggregate(reps)

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output

    # Different aggregation strategies used to capture the neighboring context.
    def mean_aggregate(self, reps):
        v = reps[0]
        u = torch.mean(torch.stack(reps[1:]), dim=0)
        return u, v

    def sum_aggregate(self, reps):
        v = reps[0]
        u = torch.sum(torch.stack(reps[1:]), dim=0)
        return u, v

    def weighted_aggregate(self, reps):
        v = reps[0]
        prev_weight = 1
        for i in range(1, 5):
            weight = 0.75 * prev_weight
            reps[i] = torch.mul(reps[i], weight)
            prev_weight = prev_weight - weight
        u = torch.sum(torch.stack(reps[1:]), dim=0)
        return u, v

    def max_aggregate(self, reps):
        v = reps[0]
        max_tensor = reps[1]
        for tensor in reps[2:]:
            max_tensor = torch.max(max_tensor, tensor)
        u = max_tensor
        return u, v
