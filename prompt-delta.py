import argparse
import math
import os
from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset \
        import YahooProcessor, SST2Processor, MnliProcessor, AgnewsProcessor
import torch
from openprompt.data_utils.utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, SoftTemplate
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils.data_sampler import FewShotSampler
from transformers import get_linear_schedule_with_warmup
from opendelta import AutoDeltaModel
from opendelta import AdapterModel, BitFitModel, LoraModel, SoftPromptModel, PrefixModel
from openprompt import PromptDataLoader




PROCESSER = {
    "sst2": SST2Processor,
    "mnli": MnliProcessor,
    "agnews": AgnewsProcessor,
    "yahoo_answers_topics": YahooProcessor,
}

MODEL_PATH = {
    "t5": "t5-base",
    "roberta": "roberta-base",
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

LR = {
    "adapter": 1e-5,
    "soft": 5e-3
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
    # torch.backends.cudnn.deterministic = True


def main(args):
    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    method = args.method
    seed = args.seed
    parameter = args.parameter
    
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)
    if method == "adapter":
        delta_model = AdapterModel(backbone_model=plm, bottleneck_dim=parameter)
        mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/TextClassification/{dataset_name}/manual_template.txt", choice=0)
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"scripts/TextClassification/{dataset_name}/manual_verbalizer.txt")
    elif method == "soft":
        delta_model = SoftPromptModel(backbone_model=plm, soft_token_num=parameter)
        mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=parameter).from_file(f"scripts/TextClassification/{dataset_name}/soft_template.txt", choice=0)
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"scripts/TextClassification/{dataset_name}/manual_verbalizer.txt")
    
    
    delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
    delta_model.log() 


    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=16,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=16,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")

    
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR[method])

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
    
    # test

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

    if method == "adapter":
        os.makedirs(f"./results/delta-adapter/{dataset_name}/{model_name}/{parameter}-dim/{seed}", exist_ok=True)
        np.save(f"./results/delta-adapter/{dataset_name}/{model_name}/{parameter}-dim/{seed}/alllabels.npy", alllabels)
        np.save(f"./results/delta-adapter/{dataset_name}/{model_name}/{parameter}-dim/{seed}/allprobs.npy", allprobs)
        np.save(f"./results/delta-adapter/{dataset_name}/{model_name}/{parameter}-dim/{seed}/allpreds.npy", allpreds)
    elif method == "soft":
        os.makedirs(f"./results/delta-soft/{dataset_name}/{model_name}/{parameter}-token/{seed}", exist_ok=True)
        np.save(f"./results/delta-soft/{dataset_name}/{model_name}/{parameter}-token/{seed}/alllabels.npy", alllabels)
        np.save(f"./results/delta-soft/{dataset_name}/{model_name}/{parameter}-token/{seed}/allprobs.npy", allprobs)
        np.save(f"./results/delta-soft/{dataset_name}/{model_name}/{parameter}-token/{seed}/allpreds.npy", allpreds)

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print('acc:', acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="roberta")
    parser.add_argument('--dataset_name', type=str, default="sst2")
    parser.add_argument('--method', type=str, default="adapter", choices=["adapter", "soft"])
    parser.add_argument('--parameter', type=int, default=0)
    
    args = parser.parse_args()
    
    if args.dataset_name == "yahoo":
        args.dataset_name = "yahoo_answers_topics"

    args.model_path = MODEL_PATH[args.model_name]
    args.dataset_path = DATASET_PATH[args.dataset_name]

    
    args.num_classes = NUM_CLASSES[args.dataset_name]
    processer = PROCESSER[args.dataset_name]()
    dataset = {}
    dataset['train'] = processer.get_train_examples(args.dataset_path)
    dataset['test'] = processer.get_test_examples(args.dataset_path)


    if args.method == "adapter":
        args.bottleneck_dim = args.parameter
    elif args.method == "soft":
        args.soft_token_num = args.parameter
    
    for i in range(args.repeats):
        set_seed(i)
        args.seed = i
        main(args)
