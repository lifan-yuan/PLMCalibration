
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor


class MnliProcessor(DataProcessor):
    # TODO Test needed
    def __init__(self):
        super().__init__()
        # self.labels = ["entailment", "neutral", "contradiction"]
        self.labels = ["No", "Yes", "Maybe"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):

                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = InputExample(
                    guid=str(idx), text_a=text_a, text_b=text_b, label=int(label)-1)
                examples.append(example)

        return examples



class AgnewsProcessor(DataProcessor):
    """
    `AG News <https://arxiv.org/pdf/1509.01626.pdf>`_ is a News Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "agnews"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["World", "Sports", "Business", "Tech"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label)-1)
                examples.append(example)
        return examples


class DBpediaProcessor(DataProcessor):
    """
    `Dbpedia <https://aclanthology.org/L16-1532.pdf>`_ is a Wikipedia Topic Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "dbpedia"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 14
        assert len(trainvalid_dataset) == 560000
        assert len(test_dataset) == 70000
    """

    def __init__(self):
        super().__init__()
        self.labels = ["company", "school", "artist", "athlete", "politics", "transportation", "building", "river", "village", "animal", "plant", "album", "film", "book",]

    def get_examples(self, data_dir, split):
        examples = []
        label_file  = open(os.path.join(data_dir,"{}_labels.txt".format(split)),'r')
        labels  = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                text_a, text_b = splited[0], splited[1:]
                text_a = text_a+"."
                text_b = ". ".join(text_b)
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(labels[idx]))
                examples.append(example)
        return examples


class ImdbProcessor(DataProcessor):
    """
    `IMDB <https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf>`_ is a Movie Review Sentiment Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "imdb"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 2
        assert len(trainvalid_dataset) == 25000
        assert len(test_dataset) == 25000
    """

    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive"]

    def get_examples(self, data_dir, split):
        examples = []
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r')
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir, '{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                example = InputExample(guid=str(idx), text_a=text_a, label=int(labels[idx]))
                examples.append(example)
        return examples


    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        label_file  = open(os.path.join(data_dir,dirname,"{}_labels.txt".format('test')),'r')
        labels  = [int(x.strip()) for x in label_file.readlines()]
        return labels



class AmazonProcessor(DataProcessor):
    """
    `Amazon <https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf>`_ is a Product Review Sentiment Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples: # TODO implement this
    """

    def __init__(self):
        super().__init__()
        self.labels = ["bad", "good"]

    def get_examples(self, data_dir, split):
        examples = []
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r')
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                example = InputExample(guid=str(idx), text_a=text_a, label=int(labels[idx]))
                examples.append(example)
        return examples


class YahooProcessor(DataProcessor):
    """
    Yahoo! Answers Topic Classification Dataset

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"
    """

    def __init__(self):
        super().__init__()
        self.labels = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music"
                        ,"Family & Relationships", "Politics & Government"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0: continue
                _, label, question_title, question_body, answer = row
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')
                try:
                    example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label))
                except:
                    print(row)
                    if label == "topic":
                        continue
                    else:
                        exit()
                examples.append(example)
        return examples


class SST2Processor(DataProcessor):
    """
    `SST-2 <https://nlp.stanford.edu/sentiment/index.html>`_ dataset is a dataset for sentiment analysis. It is a modified version containing only binary labels (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) on top of the original 5-labeled dataset released first in `Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank <https://aclanthology.org/D13-1170.pdf>`_

    We use the data released in `Making Pre-trained Language Models Better Few-shot Learners (Gao et al. 2020) <https://arxiv.org/pdf/2012.15723.pdf>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.lmbff_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "SST-2"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 2
        assert processor.get_labels() == ['0','1']
        assert len(train_dataset) == 6920
        assert len(dev_dataset) == 872
        assert len(test_dataset) == 1821
        assert train_dataset[0].text_a == 'a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films'
        assert train_dataset[0].label == 1

    """
    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive"]
        self.label_mapping = {"negative":0, "positive":1}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text_a, label=int(label))
                examples.append(example)
        return examples


class AmazonFoodProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ["bad", "good", "neutral"]
        self.label_mapping = {"bad":0, "good":1, "neutral":2}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text_a, label=int(label))
                examples.append(example)
        return examples


class CivilCommentsProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ["benign", "toxic"]
        self.label_mapping = {"benign":0, "toxic":1}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = "%s-%s" % (split, idx)
                try:
                    if int(label) != 0 and int(label) != 1:
                        print(linelist)
                        continue
                    example = InputExample(guid=guid, text_a=text_a, label=int(label))
                except:
                    continue
                examples.append(example)
        return examples




class YelpProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0: continue
                _, label, text = row
                text_a = text
                example = InputExample(guid=str(idx), text_a=text_a, label=int(label))
                examples.append(example)
        return examples


class RaceProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ["A", "B", "C", "D"]
        self.label_mappig = {"A":0, "B":1, "C":2, "D":3}

    def get_examples(self, data_dir, split):
        from datasets import load_from_disk
        dataset = load_from_disk(os.path.join(data_dir, split))
        examples = []
        for idx, data in enumerate(dataset):
            if idx == 0: continue
            article, answer, question, options = data['article'], data['answer'], data['question'], data['options']
            text_a = "Article: {}\nQuestion: {}\n".\
                        format(article.replace("\\n", " ").replace("\\", ""), question)
            text_b = "Options:\nA: {}.\nB: {}.\nC: {}.\nD: {}.\n".\
                        format(options[0], options[1], options[2], options[3])
            example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b,
                                    label=self.label_mappig[answer])
            examples.append(example)
        return examples


class SwagProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ["(a)", "(b)", "(c)", "(d)"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0: continue
                _, startphrase, sent1, sent2, gold_source, ending0, ending1, ending2, ending3, label = row
                text_a = ' '.join([sent1.replace('\\n', ' ').replace('\\', ' '),
                                   sent2.replace('\\n', ' ').replace('\\', ' ')])
                text_b = "(a): {}\n(b): {}\n(c): {}\n(d): {}\n".format(ending0, ending1, ending2, ending3)
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label))
                examples.append(example)
        return examples



class MrpcProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.txt")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[3].replace('\\', ' ')
                text_b = linelist[4].replace('\\', ' ')
                label = linelist[0]
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label))
                examples.append(example)
        return examples


class WikiProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.txt")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                text_a = line.strip()
                text_b = ""
                label = -1
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label))
                examples.append(example)
        return examples




PROCESSORS = {
    "agnews": AgnewsProcessor,
    "dbpedia": DBpediaProcessor,
    "amazon" : AmazonProcessor,
    "imdb": ImdbProcessor,
    "sst-2": SST2Processor,
    "amazon_food": AmazonFoodProcessor,
    "civil_comments": CivilCommentsProcessor,
    "mnli": MnliProcessor,
    "yahoo": YahooProcessor,
    "yelp": YelpProcessor,
    "race": RaceProcessor,
    "swag": SwagProcessor,
    "wiki": WikiProcessor,
}
