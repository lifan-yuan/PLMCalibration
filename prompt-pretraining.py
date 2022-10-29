import argparse
import math
import os
from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import AmazonFoodProcessor, CivilCommentsProcessor
from openprompt.data_utils.fewglue_dataset import RteProcessor
import torch
from openprompt.data_utils.utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils.data_sampler import FewShotSampler
from transformers import get_linear_schedule_with_warmup
from openprompt import PromptDataLoader
import torch.multiprocessing as mp

PROCESSER = {
    "amazon_food": AmazonFoodProcessor,
    "civil_comments": CivilCommentsProcessor,
}

MODEL_PATH = {
    "t5": "t5-base",
    "roberta": "roberta-base",
}

DATASET_PATH = {
    "amazon_food": "./dataset/TextClassification/amazon_food",
    "civil_comments": "./dataset/TextClassification/civil_comments",
}

NUM_CLASSES = {
    "amazon_food": 3,
    "civil_comments": 2,
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
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    seed = args.seed
    device = args.device
    mode = args.mode

    processer = PROCESSER[dataset_name]()
    dataset = {}
    dataset['train'] = processer.get_train_examples(dataset_path)
    dataset['test'] = processer.get_test_examples(dataset_path)
    print(len(dataset["train"]))
    print(model_name.split("-")[0])
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], model_path)    

    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/TextClassification/{dataset_name}/manual_template.txt", choice=0)
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"scripts/TextClassification/{dataset_name}/manual_verbalizer.txt")
    
    from transformers import T5ForConditionalGeneration, RobertaForMaskedLM
    if mode == "random": # random initialization
        if model_name == "t5":
            plm = T5ForConditionalGeneration(config=model_config).to(device)
        elif model_name == "roberta":
            plm = RobertaForMaskedLM(config=model_config).to(device)
    
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
            batch_size=16,shuffle=True, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
            batch_size=8,shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head")
    

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)

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
    allprobs = []
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        inputs = inputs.cuda(0)
        logits = prompt_model(inputs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allprobs.extend([max(prob.cpu().tolist()) for prob in probs])
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    os.makedirs(f"./results/pretraining/{dataset_name}/{model_name}-{mode}", exist_ok=True)
    np.save(f"./results/pretraining/{dataset_name}/{model_name}-{mode}/alllabels.npy", alllabels)
    np.save(f"./results/pretraining/{dataset_name}/{model_name}-{mode}/allprobs.npy", allprobs)
    np.save(f"./results/pretraining/{dataset_name}/{model_name}-{mode}/allpreds.npy", allpreds)

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print('acc:', acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="roberta")
    parser.add_argument('--dataset_name', type=str, default="sst2")
    parser.add_argument('--mode', type=str, default="pretrain", choices=["pretrain", "random"])
    args = parser.parse_args()

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.dataset_name == "yahoo":
        args.dataset_name = "yahoo_answers_topics"

    args.model_path = MODEL_PATH[args.model_name] 
    args.dataset_path = DATASET_PATH[args.dataset_name]
    args.num_classes = NUM_CLASSES[args.dataset_name]

    for i in range(args.repeats):
        set_seed(i)
        args.seed = i
        main(args)
