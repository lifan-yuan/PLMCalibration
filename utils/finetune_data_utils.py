import os
import csv



class Mnli():
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, premise, hypothesis = row
                if not str(label).isdigit():
                    continue
                examples.append(((premise, hypothesis, int(label)-1)))
        return examples



class Yahoo():
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
                _, label, question_title, question_body, answer = row
                if not str(label).isdigit():
                    continue
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
                if not str(label).isdigit():
                    continue
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
                if not str(label).isdigit():
                    continue
                examples.append((text_a, text_b, int(label)-1))
        return examples

