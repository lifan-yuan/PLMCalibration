import argparse
import os
import math
import csv
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, T5ForConditionalGeneration
from utils.finetune_data_utils import *
from utils.lstm_models import LSTMForTwoSeq



PROCESSER = {
    "yahoo_answers_topics": Yahoo,
    "mnli": Mnli,
    "sst2": SST2,
    "agnews": Agnews,
}

DATASET_PATH = {
    "sst2": "./datasets/TextClassification/SST-2",
    "mnli": "./datasets/TextClassification/mnli",
    "agnews": "./datasets/TextClassification/agnews",
    "yahoo_answers_topics": "./datasets/TextClassification/yahoo_answers_topics",
}

NUM_CLASSES = {
    "sst2": 2,
    "mnli": 3,
    "agnews": 4,
    "yahoo_answers_topics": 10,
}

# LR = {
#     "lstm": 1e-3,      
#     "roberta": 1e-5,
#     "bert": 1e-5,
#     "t5": 1e-5
# }

import random
import numpy as np
def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
          


def main(args):
    num_classes = args.num_classes
    model_name = args.model_name
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path

    processer = PROCESSER[dataset_name]()
    dataset = {}
    dataset['train'] = processer.get_examples(dataset_path, "train")
    dataset['test'] = processer.get_examples(dataset_path, "test")
    print(dataset_name)
    print(len(dataset['train']))

    
    model = LSTMForTwoSeq(output_class=num_classes).to(device)
    
    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=16)
    test_dataloader = DataLoader(dataset["test"], shuffle=False, batch_size=16)

    from transformers import get_linear_schedule_with_warmup
    loss_func = torch.nn.CrossEntropyLoss()

    optimizer_grouped_parameters = model.parameters()
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-3)
    warm_up_epochs = 3
    epochs = 5
    train_length = len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warm_up_epochs * train_length,
                                                num_training_steps=(warm_up_epochs+epochs) * train_length)

    # train
    
    model.train()
    
    for epoch in range(epochs):
        tot_loss = 0
        for step, batch in enumerate(train_dataloader):
            text_a, text_b, labels = batch
            logits = model([[sent1, sent2] for sent1, sent2 in zip(text_a, text_b)]).logits
            loss = loss_func(logits, labels.to(device))
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step %100 == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

    # test

    model.eval()
    
    allprobs = []
    allpreds = []
    alllabels = []
    for step, batch in enumerate(test_dataloader):
        text_a, text_b, labels = batch
        logits = model([[sent1, sent2] for sent1, sent2 in zip(text_a, text_b)]).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        alllabels.extend(list(labels))
        allprobs.extend([max(prob.cpu().tolist()) for prob in probs])
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    # record
    os.makedirs(f"./results/pretrain/{dataset_name}/{model_name}", exist_ok=True)
    np.save(f"./results/pretrain/{dataset_name}/{model_name}/alllabels.npy", alllabels)
    np.save(f"./results/pretrain/{dataset_name}/{model_name}/allprobs.npy", allprobs)
    np.save(f"./results/pretrain/{dataset_name}/{model_name}/allpreds.npy", allpreds)
    
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print('acc:', acc)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="sst2")
    args = parser.parse_args()
    device = torch.device("cuda")
    args.model_name = "lstm"
    
    if args.dataset_name == "yahoo":
        args.dataset_name = "yahoo_answers_topics"

    args.dataset_path = DATASET_PATH[args.dataset_name]

    args.num_classes = NUM_CLASSES[args.dataset_name]

    acc_list = []
    ECE_list = []
    prob_distribution = []
    for i in range(args.repeats):
        set_seed(i)
        args.seed = i
        main(args)
