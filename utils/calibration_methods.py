import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import random
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.svm import SVC
# from cuml.svm import SVC
from sklearn.ensemble  import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

from openprompt import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualTemplate, ManualVerbalizer

from copy import deepcopy

def wrap_feature_based_testloader(prompt_model, test_dataloader):
    feature_dataset, _, _, _, _ = compute_hidden(prompt_model, test_dataloader)
    return DataLoader(feature_dataset, shuffle=False, batch_size=16)


def wrap_verbalized_testloader(prompt_model, test_dataloader, testset, calibartion_template, tokenizer, WrapperClass):
    _, _, _, allpreds, alllabels = compute_hidden(prompt_model, test_dataloader)
    label_words = prompt_model.verbalizer.label_words
    if testset[0].text_b == "":
        for i, (pred, label) in enumerate(zip(allpreds, alllabels)):
            testset[i].text_b = label_words[pred][0]
            testset[i].label = int(pred == label)
    else:
        for i, (pred, label) in enumerate(zip(allpreds, alllabels)):
            testset[i].text_a = "(1) {} (2) {}".format(testset[i].text_a, testset[i].text_b)
            testset[i].text_b = label_words[pred][0]
            testset[i].label = int(pred == label)
    return PromptDataLoader(dataset=testset, template=calibartion_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                batch_size=16, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                truncate_method="head")


def compute_hidden(prompt_model, dataloader):
    print("Forwarding")

    prompt_model.eval()
    
    all_hidden_states = []
    allpreds = []
    alllabels = []
    with torch.no_grad():
        try:
            for step, inputs in enumerate(dataloader):
                inputs = inputs.cuda()

                output = prompt_model.prompt_model(inputs)

                batch_size = output.hidden_states[-1].shape[0]
                hidden_states = output.hidden_states[-1].reshape((batch_size, -1))
                all_hidden_states.extend(hidden_states.detach().cpu())

                logits = prompt_model(inputs).detach().cpu()
                allpreds.extend(torch.argmax(logits, dim=-1).cpu())

                labels = inputs['label']
                alllabels.extend(labels.cpu())
            dim = output.hidden_states[-1].shape[1]*output.hidden_states[-1].shape[2]
        except:
            for step, inputs in enumerate(dataloader):
                inputs = inputs.cuda()

                output = prompt_model.prompt_model(inputs)

                batch_size = output.decoder_hidden_states[-1].shape[0]
                hidden_states = output.decoder_hidden_states[-1].reshape((batch_size, -1))
                    
                all_hidden_states.extend(hidden_states.detach().cpu())

                logits = prompt_model(inputs).detach().cpu()
                allpreds.extend(torch.argmax(logits, dim=-1).cpu())

                labels = inputs['label']
                alllabels.extend(labels.cpu())
            dim = output.decoder_hidden_states[-1].shape[1]*output.decoder_hidden_states[-1].shape[2]
    
    dataset = list(zip(all_hidden_states, allpreds, alllabels))
    random.shuffle(dataset)
    
    feature_dataset = []
    for hidden_states, pred, label in dataset:
        feature_dataset.append((hidden_states, int(pred==label)))
    
    prompt_model.train()

    return feature_dataset, dim, all_hidden_states, allpreds, alllabels




def feature_based(prompt_model, dev_dataloader):

    # dataset
    feature_dataset, dim, _, _, _ = compute_hidden(prompt_model, dev_dataloader)
    train_dataloader = DataLoader(feature_dataset, shuffle=True, batch_size=16)
    # model    
    classifier = torch.nn.Linear(dim, 2).cuda()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()
    
    print("Training Calibrater")
    prompt_model.train()
    for epoch in range(10):
        tot_loss = 0
        for step, batch in enumerate(train_dataloader):
            inputs, labels = batch
            logits = classifier(inputs.cuda())

            loss = loss_func(logits, labels.cuda())
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

    return classifier
    

def verbalized(prompt_model, train_dataloader, dev_dataloader, devset, dataset_name, tokenizer, WrapperClass):
    # 1. get prediction
    _, _, _, allpreds, alllabels = compute_hidden(prompt_model, dev_dataloader)
    # 2. relabeling
    label_words = prompt_model.verbalizer.label_words
    label_count = {0:0, 1:0}
    if devset[0].text_b == "":
        for i, (pred, label) in enumerate(zip(allpreds, alllabels)):
            devset[i].text_b = label_words[pred][0]
            devset[i].label = int(pred == label)
            label_count[int(pred == label)] += 1
    else:
        for i, (pred, label) in enumerate(zip(allpreds, alllabels)):
            devset[i].text_a = "(1) {} (2) {}".format(devset[i].text_a, devset[i].text_b)
            devset[i].text_b = label_words[pred][0]
            devset[i].label = int(pred == label)
            label_count[int(pred == label)] += 1
    # label balance
    print("before balance:", label_count)
    shots = min(label_count[0], label_count[1])
    print("shots:", shots)
    dataset = []
    label_count = {0:0, 1:0}
    for data in devset:
        if label_count[data.label] < shots:
            dataset.append(data)
            label_count[data.label] += 1
    devset = dataset
    print("after balance:", label_count)

    # 3.wrap
    calibartion_template = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/TextClassification/{dataset_name}/calibration_template.txt", choice=0)
    calibartion_verbalizer = ManualVerbalizer(tokenizer, num_classes=2).from_file(f"scripts/TextClassification/{dataset_name}/calibration_verbalizer.txt")
    train_dataloader = PromptDataLoader(dataset=devset, template=calibartion_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")
    # train
    classifier = PromptForClassification(plm=deepcopy(prompt_model.plm),template=calibartion_template, verbalizer=calibartion_verbalizer, freeze_plm=False).cuda()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)

    loss_func = torch.nn.CrossEntropyLoss()

    prompt_model.train()

    for epoch in range(10):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            inputs = inputs.cuda()
            logits = classifier(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

    return classifier, calibartion_template, calibartion_verbalizer


def verbalized_iterative(prompt_model, train_dataloader, dev_dataloader, devset, dataset_name, tokenizer, WrapperClass):
    # 1. get prediction
    _, _, _, allpreds, alllabels = compute_hidden(prompt_model, dev_dataloader)
    # 2. relabeling
    label_words = prompt_model.verbalizer.label_words
    label_count = {0:0, 1:0}
    if devset[0].text_b == "":
        for i, (pred, label) in enumerate(zip(allpreds, alllabels)):
            devset[i].text_b = label_words[pred][0]
            devset[i].label = int(pred == label)
            label_count[int(pred == label)] += 1
    else:
        for i, (pred, label) in enumerate(zip(allpreds, alllabels)):
            devset[i].text_a = "(1) {} (2) {}".format(devset[i].text_a, devset[i].text_b)
            devset[i].text_b = label_words[pred][0]
            devset[i].label = int(pred == label)
            label_count[int(pred == label)] += 1
    # label balance
    print("before balance:", label_count)
    shots = min(label_count[0], label_count[1])
    print("shots:", shots)
    dataset = []
    label_count = {0:0, 1:0}
    for data in devset:
        if label_count[data.label] < shots:
            dataset.append(data)
            label_count[data.label] += 1
    devset = dataset
    print("after balance:", label_count)

    # 3.wrap
    performance_template = prompt_model.template
    performance_verbalizer = prompt_model.verbalizer
    calibartion_template = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/TextClassification/{dataset_name}/calibration_template.txt", choice=0)
    calibartion_verbalizer = ManualVerbalizer(tokenizer, num_classes=2).from_file(f"scripts/TextClassification/{dataset_name}/calibration_verbalizer.txt")
    train_dataloader_performance = train_dataloader # balanced
    train_dataloader_calibration = PromptDataLoader(dataset=devset, template=calibartion_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)

    loss_func = torch.nn.CrossEntropyLoss()

    prompt_model.train()

    for epoch in range(10):
        tot_loss = 0

        print("Training model for better calibration")
        prompt_model = PromptForClassification(plm=prompt_model.plm, template=calibartion_template, verbalizer=calibartion_verbalizer, freeze_plm=False).cuda()
        for step, inputs in enumerate(train_dataloader_calibration):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print("Calibration Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)


        print("Training model for better performance")
        prompt_model = PromptForClassification(plm=prompt_model.plm, template=performance_template, verbalizer=performance_verbalizer, freeze_plm=False).cuda()
        for step, inputs in enumerate(train_dataloader_performance):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print("Performance Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

        

    return prompt_model, calibartion_template, calibartion_verbalizer


def verbalized_multitask(prompt_model, train_dataloader, dev_dataloader, devset, dataset_name, tokenizer, WrapperClass):
    # 1. get prediction
    _, _, _, allpreds, alllabels = compute_hidden(prompt_model, dev_dataloader)
    # 2. relabeling
    label_words = prompt_model.verbalizer.label_words
    label_count = {0:0, 1:0}
    if devset[0].text_b == "":
        for i, (pred, label) in enumerate(zip(allpreds, alllabels)):
            devset[i].text_b = label_words[pred][0]
            devset[i].label = int(pred == label)
            label_count[int(pred == label)] += 1
    else:
        for i, (pred, label) in enumerate(zip(allpreds, alllabels)):
            devset[i].text_a = "(1) {} (2) {}".format(devset[i].text_a, devset[i].text_b)
            devset[i].text_b = label_words[pred][0]
            devset[i].label = int(pred == label)
            label_count[int(pred == label)] += 1
    # label balance
    print("before balance:", label_count)
    shots = min(label_count[0], label_count[1])
    print("shots:", shots)
    dataset = []
    label_count = {0:0, 1:0}
    for data in devset:
        if label_count[data.label] < shots:
            dataset.append(data)
            label_count[data.label] += 1
    devset = dataset
    print("after balance:", label_count)
    
    # 3.wrap
    performance_template = prompt_model.template
    performance_verbalizer = prompt_model.verbalizer
    calibartion_template = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/TextClassification/{dataset_name}/calibration_template.txt", choice=0)
    calibartion_verbalizer = ManualVerbalizer(tokenizer, num_classes=2).from_file(f"scripts/TextClassification/{dataset_name}/calibration_verbalizer.txt")
    train_dataloader_performance = train_dataloader
    train_dataloader_calibration = PromptDataLoader(dataset=devset, template=calibartion_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)

    loss_func = torch.nn.CrossEntropyLoss()

    prompt_model_calibration = PromptForClassification(plm=prompt_model.plm, template=calibartion_template, verbalizer=calibartion_verbalizer, freeze_plm=False).cuda()
    prompt_model_performance = PromptForClassification(plm=prompt_model.plm, template=performance_template, verbalizer=performance_verbalizer, freeze_plm=False).cuda()

    max_step = max(len(train_dataloader_calibration), len(train_dataloader_performance))
    print(f"{max_step=}")
    calibration_loader = iter(train_dataloader_calibration)
    performance_loader = iter(train_dataloader_performance)

    prompt_model.train()

    for epoch in range(10):
        tot_calibration_loss = 0
        tot_performance_loss = 0
        for step in range(max_step):

            try:
                calibration_inputs = next(calibration_loader)
            except StopIteration:
                train_dataloader_calibration = PromptDataLoader(dataset=devset, template=calibartion_template, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                    batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                    truncate_method="tail")
                calibration_loader = iter(train_dataloader_calibration)
                calibration_inputs = next(calibration_loader)

            calibration_inputs = calibration_inputs.cuda()
            calibration_logits = prompt_model_calibration(calibration_inputs)
            calibration_labels = calibration_inputs['label']
            calibration_loss = loss_func(calibration_logits, calibration_labels)

            try:
                performance_inputs = next(performance_loader)
            except StopIteration:
                train_dataloader_performance = train_dataloader
                performance_loader = iter(train_dataloader_performance)
                performance_inputs = next(performance_loader)
                
            performance_inputs = performance_inputs.cuda()
            performance_logits = prompt_model_performance(performance_inputs)
            performance_labels = performance_inputs['label']
            performance_loss = loss_func(performance_logits, performance_labels)

            total_loss = calibration_loss + performance_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tot_calibration_loss += calibration_loss.item()
            tot_performance_loss += performance_loss.item()
            
            if step % 100 == 1:
                print("Epoch {}, average calibration loss: {}, average performance loss: {}"\
                        .format(epoch, tot_calibration_loss/(step+1), tot_performance_loss/(step+1)), flush=True)

    return prompt_model, calibartion_template, calibartion_verbalizer










class Temperature(nn.Module):
    def __init__(self):
        super(Temperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


def one_hot_tensor(y_batch_tensor, num_classes):
    y_tensor = torch.FloatTensor(y_batch_tensor.size(0), num_classes).fill_(0).cuda()
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor


def nll_criterion(output, target, num_classes=3):
    target = one_hot_tensor(target, num_classes)
    prob_true = output.mul(target).sum(dim=-1).flatten()
    return -prob_true.log().mean()


def temperature_scaling(dev_dataloader, prompt_model):

    prompt_model.eval()

    alllogits = []
    alllabels = []
    # get logits
    for step, inputs in enumerate(dev_dataloader):
        inputs = inputs.cuda()
        logits = prompt_model(inputs).detach().cpu()
        labels = inputs['label'].detach().cpu()
        alllogits.append(logits)
        alllabels.append(labels)
    
    alllogits = torch.cat(alllogits).cuda()
    alllabels = torch.cat(alllabels).cuda()
    num_classes = len(set(alllabels.tolist()))
    # tune T
    ########################### Grid Search ############################
    # best_nll = float("inf")
    # temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))
    # i = 0
    # for temp in tqdm(temp_values, leave=False, desc='training'):
    #     nll = nll_criterion(F.softmax(alllogits / temp, dim=-1), alllabels, i)
    #     # nll = nll_criterion(alllogits / temp, alllabels)
    #     if temp % 0.5 == 0:
    #         print(alllogits)
    #         print(F.softmax(alllogits / temp, dim=-1))
    #         print(alllabels)
    #         print(temp)
    #         print("nll", nll)
    #         print("best_nll", best_nll)
    #     if nll < best_nll:
    #         # print("update")
    #         # print(nll.item(), best_nll)
    #         best_nll = nll.item()
    #         best_temp = temp
    #     i += 1
    ########################### Grid Search ############################

    ######################## AdamW optimization ########################
    temp_scaling = Temperature().cuda()
    max_iter = 5000
    optimizer = torch.optim.AdamW(temp_scaling.parameters(), lr=1e-2)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=max_iter * len(alllogits))

    nll_list = []
    T_list = []
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = nll_criterion(F.softmax(temp_scaling(alllogits), dim=-1), alllabels, num_classes)
        nll_list.append(loss.detach().cpu().item())
        T_list.append(temp_scaling.temperature.data.detach().cpu().item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    best_temp = temp_scaling.temperature.data.detach().cpu().numpy()
    ######################## AdamW optimization ########################

    ######################## LBFGS optimization ########################
    # temp_scaling = Temperature().cuda()
    # max_iter = 5000
    # nll_criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.LBFGS(temp_scaling.parameters(), lr=1e-2, max_iter=max_iter)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0,
    #                                             num_training_steps=max_iter * len(alllogits))
    # with torch.no_grad():                                                
    #     before = nll_criterion(temp_scaling(alllogits), alllabels)
    # def eval():
    #     optimizer.zero_grad()
    #     loss = nll_criterion(temp_scaling(alllogits), alllabels)
    #     loss.backward()
    #     scheduler.step()
    #     return loss

    # optimizer.step(eval)
    # with torch.no_grad():                                                
    #     after = nll_criterion(temp_scaling(alllogits), alllabels)
    # print(f"{before=}")
    # print(f"{after=}")
    # best_temp = temp_scaling.temperature.data.detach().cpu().numpy()
    ######################## LBFGS optimization ########################
    
    print(f"{best_temp=}")

    prompt_model.train()

    return best_temp



class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss. Adapted from https://bit.ly/2T6kfz7. If 0 < smoothing < 1,
    this smoothes the standard cross-entropy loss.
    """

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1. - smoothing
        smoothing_value = smoothing / (num_classes - 1)
        one_hot = torch.full((num_classes,), smoothing_value).cuda()
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, output, target):
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(F.log_softmax(output, 1), model_prob, reduction='sum')



# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
import os
import csv
from openprompt.data_utils.utils import InputExample


class EDASST2Processor():

    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive"]
        self.label_mapping = {"negative":0, "positive":1}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        data = pd.read_csv(path, sep='\t').values.tolist()
        examples = []
        for idx1, item in enumerate(data):
            if not np.isnan(item[1]):
                text_a = item[0].strip()
                label = item[1]
                try:
                    if not str(label).isdigit():
                        continue
                    augmented_sentences = eda(text_a)
                except:
                    continue
                for idx2, aug_sent in enumerate(augmented_sentences):
                    guid = f"{split}-{idx1}-{idx2}"
                    example = InputExample(guid=guid, text_a=aug_sent, label=int(label))
                    examples.append(example)

        return examples


class EDAMnliProcessor():
    # TODO Test needed
    def __init__(self):
        super().__init__()
        # self.labels = ["contradiction", "entailment", "neutral"]
        self.labels = ["entailment", "neutral", "contradiction"]
        # self.labels = ["No", "Maybe", "Yes"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx1, row in enumerate(reader):
                label, premise, hypothesis = row
                try:
                    if not str(label).isdigit():
                        continue
                    augmented_sentences = eda(hypothesis)
                except:
                    continue
                for idx2, aug_sent in enumerate(augmented_sentences):
                    guid = f"{split}-{idx1}-{idx2}"
                    text_a = premise.replace('\\', ' ')
                    text_b = aug_sent.replace('\\', ' ')
                    example = InputExample(
                        guid=guid, text_a=text_a, text_b=text_b, label=int(label)-1)
                    examples.append(example)

        return examples


class EDAAmazonFoodProcessor():

    def __init__(self):
        super().__init__()
        self.labels = ["bad", "good", "neutral"]
        self.label_mapping = {"bad":0, "good":1, "neutral":2}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        data = pd.read_csv(path, sep='\t').values.tolist()
        examples = []
        for idx1, item in enumerate(data):
            text_a = item[0].strip()
            label = item[1]
            try:
                if not str(label).isdigit():
                    continue
                augmented_sentences = eda(text_a)
            except:
                print(text_a)
                continue
            for idx2, aug_sent in enumerate(augmented_sentences):
                guid = f"{split}-{idx1}-{idx2}"
                example = InputExample(guid=guid, text_a=aug_sent, label=int(label))
                examples.append(example)
        return examples


class EDACivilCommentsProcessor():

    def __init__(self):
        super().__init__()
        self.labels = ["benign", "toxic"]
        self.label_mapping = {"benign":0, "toxic":1}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        data = pd.read_csv(path, sep='\t').values.tolist()
        examples = []
        for idx1, item in enumerate(data):
            text_a = item[0].strip()
            label = item[1]
            try:
                if not str(label).isdigit():
                    continue
                augmented_sentences = eda(text_a)
            except:
                print(text_a)
                continue
            for idx2, aug_sent in enumerate(augmented_sentences):
                guid = f"{split}-{idx1}-{idx2}"
                example = InputExample(guid=guid, text_a=aug_sent, label=int(label))
                examples.append(example)
                
        return examples


class EDAYahooProcessor():

    def __init__(self):
        super().__init__()
        self.labels = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music"
                        ,"Family & Relationships", "Politics & Government"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx1, row in enumerate(reader):
                if idx1 == 0: continue
                _, label, question_title, question_body, answer = row
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')
                
                try:
                    if not str(label).isdigit():
                        continue
                    augmented_sentences = eda(text_a)
                except:
                    print(text_a)
                    print(label)
                    continue
                for idx2, aug_sent in enumerate(augmented_sentences):
                    guid = f"{split}-{idx1}-{idx2}"
                    example = InputExample(guid=guid, text_a=aug_sent, text_b=text_b, label=int(label))
                    examples.append(example)
        
        return examples




import random
from random import shuffle
random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 

def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	return new_words

def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1):
	
	sentence = get_only_chars(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word != '']
	num_words = len(words)
	
	augmented_sentences = []
	num_new_per_technique = int(num_aug/4)+1

	#sr
	if (alpha_sr > 0):
		n_sr = max(1, int(alpha_sr*num_words))
		for _ in range(num_new_per_technique):
			a_words = synonym_replacement(words, n_sr)
			augmented_sentences.append(' '.join(a_words))

	#ri
	if (alpha_ri > 0):
		n_ri = max(1, int(alpha_ri*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri)
			augmented_sentences.append(' '.join(a_words))

	#rs
	if (alpha_rs > 0):
		n_rs = max(1, int(alpha_rs*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs)
			augmented_sentences.append(' '.join(a_words))

	#rd
	if (p_rd > 0):
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, p_rd)
			augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	augmented_sentences.append(sentence)

	return augmented_sentences