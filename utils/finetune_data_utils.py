import argparse
import os
import math
import csv



class Mnli():
    # TODO Test needed
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, premise, hypothesis = row
                examples.append(((premise, hypothesis, int(label)-1)))
        return examples



class Yahoo():
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
                examples.append((text_a, text_b, int(label)))
        return examples



class SST2():

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
                text_b = ""
                label = linelist[1]
                examples.append((text_a, text_b, int(label)))
        return examples


class Agnews():
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
                examples.append((text_a, text_b, int(label)-1))
        return examples



class RTE():
    def __init__(self):
        super().__init__()
        self.label_mapping = {"not_entailment":0, "entailment":1}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[1]
                text_b = linelist[2]
                label = linelist[3]
                examples.append((text_a, text_b, self.label_mapping[label]))
        return examples



class MRPC():
    def __init__(self):
        super().__init__()
        
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.txt")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[3]
                text_b = linelist[4]
                label = linelist[0]
                examples.append((text_a, text_b, int(label)))
        return examples