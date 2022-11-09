import torch.nn as nn
import pandas as pd
import abc
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import namedtuple
from transformers import BertTokenizer

class Model(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.outputs = namedtuple('outputs', ['loss', 'logits', 'pred_labels'])
        self.loss_func = torch.nn.CrossEntropyLoss()

    # get model's outputs, for users to call
    @abc.abstractmethod
    def __call__(self, sentences):
        """
        :param input: a raw sentnece or a list of raw sentences
        :return: a namedtuple('outputs', ['loss', 'probs', 'pred_labels'])
                loss: tensor, result of loss function
                probs: tensor, probability distribution for each class
                pred_labels: tensor, prediction of label for this sentence
        """
        return self.outputs(None, None, None)

class LSTM(Model, metaclass=abc.ABCMeta):
    def __init__(self, glove_path, vocab_path, embed_dim, hidden_size, layers, bidirectional, dropout, output_class):
        super().__init__()

        
        with open(vocab_path, "r") as f:
            lines = f.readlines()
            words = []
            index = []
            for idx, line in enumerate(lines):
                line = line.strip()
                words.append(line)
                index.append(idx)

        self.vocab_dict = dict(zip(words, index))

        if glove_path is None:
            self.embedding = nn.Embedding(idx+1, embed_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.load(glove_path))
            self.embedding.requires_grad_(False)

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout, )

        self.linear = None
        self.loss = torch.nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_vocab_ids(self, src):
        tokens = self.tokenizer.tokenize(src)
        tgt = self.tokenizer.convert_tokens_to_string(tokens)
        vocab_ids = []

        for word in tgt.split(' '):
            if word in self.vocab_dict.keys():
                id = self.vocab_dict[word]
            else:
                id = self.vocab_dict['[UNK]']
            vocab_ids.append(id)

        return vocab_ids

    def get_batch(self, inputs):
        input_id_list = []
        length_list = []

        if isinstance(inputs, str):
            inputs = [inputs]

        if isinstance(inputs[0], str):
            input_ids = []
            lengths = []

            for src in inputs:
                ids = self.get_vocab_ids(src)
                input_ids.append(torch.tensor(ids))
                lengths.append(len(ids))

            input_ids = pad_sequence(input_ids, batch_first=True)

            input_id_list.append(input_ids)
            length_list.append(lengths)

        elif isinstance(inputs[0], list):
            input_ids_1 = []
            lengths_1 = []
            input_ids_2 = []
            lengths_2 = []

            for src1, src2 in inputs:
                ids_1 = self.get_vocab_ids(src1)
                input_ids_1.append(torch.tensor(ids_1))
                lengths_1.append(len(ids_1))

                ids_2 = self.get_vocab_ids(src2)
                input_ids_2.append(torch.tensor(ids_2))
                lengths_2.append(len(ids_2))

            input_ids_1 = pad_sequence(input_ids_1, batch_first=True)
            input_ids_2 = pad_sequence(input_ids_2, batch_first=True)

            input_id_list += [input_ids_1, input_ids_2]
            length_list += [lengths_1, lengths_2]

        return {'input_id_list': input_id_list, 'length_list': length_list}

    def __call__(self, sentences=None, labels=None, **kwargs):
        if sentences is not None:
            inputs = self.get_batch(sentences)
        else:
            inputs = kwargs

        logits = self.forward(**inputs)
        probs = torch.softmax(logits, -1)
        pred_labels = torch.argmax(probs, -1)

        if labels is None:
            loss = None
        else:
            labels = torch.tensor(labels).to(next(self.parameters()).device)
            loss = self.loss(logits, labels)

        return self.outputs(loss, logits, pred_labels)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    @abc.abstractmethod
    def forward(self, **kwargs):
        raise Exception('function "forward" is an abstract function!')


class LSTMForTwoSeq(LSTM):
    def __init__(self,
                 glove_path=None,
                 vocab_path="./utils/vocab.txt",
                 embed_dim=300,
                 hidden_size=1024,
                 layers=2,
                 bidirectional=True,
                 dropout=0.1,
                 output_class=2):
        super().__init__(glove_path, vocab_path, embed_dim, hidden_size, layers, bidirectional, dropout, output_class)

        linear_size = 8 * hidden_size if bidirectional else 4 * hidden_size
        self.linear = torch.nn.Linear(linear_size, output_class)

    def encode(self, input_ids, lengths):
        texts_embedding = self.embedding(input_ids)
        packed_inputs = pack_padded_sequence(texts_embedding, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-2, :, :]
        backward_hidden = hn[-1, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        return concat_hidden

    def forward(self, input_id_list, length_list):
        if self.training:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

        input_ids_1 = input_id_list[0].to(next(self.parameters()).device)
        input_ids_2 = input_id_list[1].to(next(self.parameters()).device)
        lengths_1 = length_list[0]
        lengths_2 = length_list[1]

        vector1 = self.encode(input_ids_1, lengths_1)
        vector2 = self.encode(input_ids_2, lengths_2)

        features = torch.cat((vector1, vector2, torch.abs(vector1 - vector2), vector1 * vector2), 1)
        output = self.linear(features)

        return output
