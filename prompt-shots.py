import argparse
import math
import os
from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset \
        import YahooProcessor, SST2Processor, AgnewsProcessor, MnliProcessor
import torch
from openprompt.data_utils.utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils.data_sampler import FewShotSampler
from transformers import get_linear_schedule_with_warmup
from openprompt import PromptDataLoader


PROCESSER = {
    "sst2": SST2Processor,
    "mnli": MnliProcessor,
    "agnews": AgnewsProcessor,
    "yahoo_answers_topics": YahooProcessor,
}

MODEL_PATH = {
    "t5": "./model_cache/t5-base",
    "roberta": "./model_cache/roberta-base",
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



import random
import numpy as np
import torch
def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    num_classes = args.num_classes
    shots = args.shots
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    seed = args.seed
    
    global dataset

    plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], model_path)
    
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/TextClassification/{dataset_name}/manual_template.txt", choice=0)
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"scripts/TextClassification/{dataset_name}/manual_verbalizer.txt")

    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
            batch_size=16,shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head")
    
    if shots > 0 or shots == -1:

        if shots > 0:
            train_sampler = FewShotSampler(num_examples_per_label=shots, also_sample_dev=False)
            trainset = train_sampler(dataset['train'], seed=seed)

        train_dataloader = PromptDataLoader(dataset=trainset, template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
            batch_size=16,shuffle=True, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")
        

        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)

        prompt_model.train()

        for epoch in range(10):
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):
                inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step %100 == 1:
                    print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
    

    prompt_model.eval()
    
    allprobs = []
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        inputs = inputs.cuda()
        logits = prompt_model(inputs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allprobs.extend([max(prob.cpu().tolist()) for prob in probs])
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    os.makedirs(f"./results/shots/{dataset_name}/{model_name}/{shots}-shots/{seed}", exist_ok=True)
    np.save(f"./results/shots/{dataset_name}/{model_name}/{shots}-shots/{seed}/alllabels.npy", alllabels)
    np.save(f"./results/shots/{dataset_name}/{model_name}/{shots}-shots/{seed}/allprobs.npy", allprobs)
    np.save(f"./results/shots/{dataset_name}/{model_name}/{shots}-shots/{seed}/allpreds.npy", allpreds)

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print('acc:', acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=-1)
    parser.add_argument('--shots', type=int, default=[0], nargs="+")
    parser.add_argument('--model_name', type=str, default="roberta")
    parser.add_argument('--dataset_name', type=str, default="sst2")
    args = parser.parse_args()
    
    if args.dataset_name == "yahoo":
        args.dataset_name = "yahoo_answers_topics"

    args.model_path = MODEL_PATH[args.model_name]
    args.dataset_path = DATASET_PATH[args.dataset_name]
    
    args.num_classes = NUM_CLASSES[args.dataset_name]

    
    processer = PROCESSER[args.dataset_name]()
    global dataset
    dataset = {}
    dataset['train'] = processer.get_train_examples(args.dataset_path)
    dataset['test'] = processer.get_test_examples(args.dataset_path)

    print(f"{len(dataset['train'])=}")
    if args.repeats == -1:
        for shots in [2**i for i in range(25)]:
            args.shots = shots
            if shots * args.num_classes > len(dataset["train"]):
                print("Too much samples!")
                print("Exit!")
                exit()

            if shots <= 0:
                repeats = 1 # full data
            elif shots < 100:
                repeats = 5
            elif shots < 1000:
                repeats = 3
            else:
                repeats = 1
            
            for i in range(repeats):
                set_seed(i)
                args.seed = i
                main(args)

    else:
        repeats = args.repeats
        for shots in args.shots:
            if shots/2 * args.num_classes > len(dataset["train"]):
                print("Too much samples!")
                print("Exit!")
                exit()

            for i in range(repeats):
                set_seed(i)
                args.seed = i
                main(args)
