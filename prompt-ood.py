import argparse
import math
import os
# In this scripts, you will laern how to do calibartion and zero-shot learning
# We use manual verbalizer and knowledgeable verbalizer as examples.
from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import *
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
from transformers import T5ForConditionalGeneration, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from calibration_methods import *

PROCESSER = {
    "sst2": SST2Processor,
    "mnli": MnliProcessor,
    "yahoo_answers_topics": YahooProcessor,
    "amazon_food": AmazonFoodProcessor,
    "civil_comments": CivilCommentsProcessor,
    "dynasent": AmazonFoodProcessor
}

EDA_PROCESSER = {
    "sst2": EDASST2Processor,
    "mnli": EDAMnliProcessor,
    "amazon_food": EDAAmazonFoodProcessor,
    "civil_comments": EDACivilCommentsProcessor,
    "dynasent": EDAAmazonFoodProcessor,
    "yahoo_answers_topics": EDAYahooProcessor
}

MODEL_PATH = {
    "t5": "./model_cache/t5-base",
    "roberta": "./model_cache/roberta-base",
}

DATASET_PATH = {
    "sst2": "./datasets/TextClassification/SST-2",
    "mnli": "./datasets/TextClassification/mnli",
    "yahoo_answers_topics": "./datasets/TextClassification/yahoo_answers_topics",
    "amazon_food": "./datasets/TextClassification/amazon_food",
    "civil_comments": "./datasets/TextClassification/civil_comments",
    "dynasent": "./datasets/TextClassification/dynasent",
}

NUM_CLASSES = {
    "sst2": 2,
    "mnli": 3,
    "yahoo_answers_topics": 10,
    "amazon_food": 3,
    "civil_comments": 2,
    "dynasent": 3
}



import random
import numpy as np
def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
          

def evaluation(test_dataloader, prompt_model, dataset_name, model_name, ood_name, method, seed):

    allprobs = []
    allpreds = []
    alllabels = []

    global T
    T = torch.tensor(T).cuda() if method == "temperature_scaling" else 1

    try:
        for step, inputs in enumerate(test_dataloader):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            probs = F.softmax(logits / T, dim=-1)
            if ood_name == "hans" and "calibration" not in method:
                probs = torch.stack([probs[:,0], probs[:,1]+probs[:,2]], dim=1)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            if "calibration" not in method:
                allprobs.extend([prob.max().item() for prob in probs])
            else: # the prob of "True"
                allprobs.extend([prob[1].item() for prob in probs])
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    except:
        for step, batch in enumerate(test_dataloader):
            inputs, labels = batch
            logits = prompt_model(torch.tensor(inputs).cuda())
            probs = F.softmax(logits, dim=-1)
            if ood_name == "hans" and "calibration" not in method:
                probs = torch.stack([probs[:,0], probs[:,1]+probs[:,2]], dim=1)
            if "calibration" not in method:
                allprobs.extend([prob.max().item() for prob in probs])
            else:
                allprobs.extend([prob[1].item() for prob in probs])
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            alllabels.extend(labels)

    # record
    os.makedirs(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}", exist_ok=True)
    np.save(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/alllabels.npy", alllabels)
    np.save(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allprobs.npy", allprobs)
    np.save(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allpreds.npy", allpreds)
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print('acc on {}: {}'.format(ood_name, acc))
    return acc


def compute_entropy(test_dataloader, prompt_model, dataset_name, model_name, ood_name, method, seed):

    allprobs = []
    allentropy = []
    global T
    T = torch.tensor(T).cuda() if method == "temperature_scaling" else 1

    try:
        for step, inputs in enumerate(test_dataloader):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            probs = F.softmax(logits / T, dim=-1)
            entropy = -probs.mul(probs.detach().clone().log()).sum(dim=-1).flatten()
            allentropy.extend(entropy.tolist())
            if "calibration" not in method:
                allprobs.extend([prob.max().item() for prob in probs])
            else: # the prob of "True"
                allprobs.extend([prob[1].item() for prob in probs])
    except:
        for step, batch in enumerate(test_dataloader):
            inputs, labels = batch
            logits = prompt_model(torch.tensor(inputs).cuda())
            probs = F.softmax(logits, dim=-1)
            entropy = -probs.mul(probs.detach().clone().log()).sum(dim=-1).flatten()
            allentropy.extend(entropy.tolist())
            if "calibration" not in method:
                allprobs.extend([prob.max().item() for prob in probs])
            else:
                allprobs.extend([prob[1].item() for prob in probs])

    # record
    os.makedirs(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}", exist_ok=True)
    np.save(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allprobs.npy", allprobs)
    np.save(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allentropy.npy", allentropy)

    avg_entropy = np.mean(allentropy)
    print('entropy on {}: {}'.format(ood_name, avg_entropy))



def eval(prompt_model, train_dataloader, devset, dev_dataloader, processer, dataset_path, ood_dataset_path, mytemplate, tokenizer, WrapperClass, dataset_name, model_name, method, seed):
    print("evaluation")
    
    OOD_NAME = {
        "mnli": ["snli", "hans", "anli"],
        "amazon_food": ["sst5", "semeval"],
        "civil_comments": ["hate_speech", "implicit_hate"],
        "sst2": ["bookcorpus", "random_words"],
        "dynasent": ["dsc", "amazon_food"],
        "yahoo_answers_topics": ["bookcorpus", "random_words"]
    }

    dataset = {}
    dataset['test'] = processer.get_examples(dataset_path, "test")
    for ood_name in OOD_NAME[dataset_name]:
        if ood_name not in ["wikitext", "bookcorpus","random_words"]:
            dataset[ood_name] = processer.get_examples(ood_dataset_path, ood_name)
        else:
            dataset[ood_name] = WikiProcessor().get_examples(f"./datasets/TextClassification/{ood_name}", ood_name)

    dataloader_dict = {}
    for ood_name in dataset.keys(): # including the test split
        test_dataloader = PromptDataLoader(dataset=dataset[ood_name], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
            batch_size=16,shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head")
        dataloader_dict[ood_name] = test_dataloader

    # evaluate performance
    if method not in ["I-PLM", "I-iterative","I-multitask"]:
        print("Performance:")
        if dataset_name not in ["sst2", "yahoo_answers_topics"]:

            if method in ["E-MLP", "E-PLM"]:
                acc_path = f"./results/metrics/ood/{dataset_name}/accuracy_of_{method}/{model_name}"
                os.makedirs(acc_path, exist_ok=True)
                                
            for ood_name, test_dataloader in dataloader_dict.items():
                if ood_name == "test":
                    ood_name = f"{dataset_name}_iid"
                acc = evaluation(test_dataloader, prompt_model, dataset_name, model_name, ood_name, method, seed)
                if method in ["E-MLP", "E-PLM"]:
                    np.save(os.path.join(acc_path, f"{ood_name}-{seed}.npy"), acc)

        else:
            for ood_name, test_dataloader in dataloader_dict.items():
                if ood_name == "test":
                    ood_name = f"{dataset_name}_iid"
                    evaluation(test_dataloader, prompt_model, dataset_name, model_name, ood_name, method, seed)
                compute_entropy(test_dataloader, prompt_model, dataset_name, model_name, ood_name, method, seed)
        
        if method not in ["E-MLP", "E-PLM"]: # i.e., unlearnable.
            print("finish evaluation")
            return

    #--------------------------------------------------------------------------------#
    #----------------------------- Learnable Calibrater -----------------------------#
    #--------------------------------------------------------------------------------#

    # evaluate calibration of learnable methods
    # load the calibrater and re-wrap the dataloader
    if method == "E-MLP":
        path = f"./model_cache/ood_{dataset_name}/{model_name}/E-MLP/{seed}"
        if os.path.exists(path):
            print("Load the calibrater")
            dim = np.load(os.path.join(path, "dim.npy"))
            calibrater = torch.nn.Linear(dim, 2).cuda()
            calibrater.load_state_dict(torch.load(os.path.join(path, "calibrater.pt")))
            
        else:
            print("Train the calibrater")
            os.makedirs(path, exist_ok=True)
            calibrater = feature_based(prompt_model, dev_dataloader)    
            dim = calibrater.in_features
            np.save(os.path.join(path, "dim.npy"), dim)
            torch.save(calibrater.state_dict(), os.path.join(path, "calibrater.pt"))
        # re-wrap for evaluating calibration
        for ood_name, test_dataloader in dataloader_dict.items():
            test_dataloader =  wrap_feature_based_testloader(prompt_model, test_dataloader)
            dataloader_dict[ood_name] = test_dataloader

    else:

        if method == "I-PLM":
            path = f"./model_cache/ood_{dataset_name}/{model_name}/E-PLM/{seed}"
        else:
            path = f"./model_cache/ood_{dataset_name}/{model_name}/{method}/{seed}"

        if os.path.exists(path):
            print(f"Load the classifier from {path}")
            plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], path)
            calibartion_template = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/TextClassification/{dataset_name}/calibration_template.txt", choice=0)
            calibartion_verbalizer = ManualVerbalizer(tokenizer, num_classes=2).from_file(f"scripts/TextClassification/{dataset_name}/calibration_verbalizer.txt")
            if method in ["I-PLM", "I-iterative", "I-multitask"]: 
                # the backbone plm of the answer-provider has been tuned on the calibration task.
                # use the original template to test the performance.
                template, verbalizer = mytemplate, prompt_model.verbalizer 
                classifier = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=False).cuda()
            else: 
                classifier = prompt_model # for "verbalized", the answer-provider is the original prompt_model
        else:
            print("Train the classifier")
            TRAINER = { "E-PLM": verbalized,
                        "I-PLM": verbalized, 
                        "I-iterative": verbalized_iterative,
                        "I-multitask": verbalized_multitask}
            trainer = TRAINER[method]
            # the returned classifier can be directly used for *performance* evaluation
            classifier, calibartion_template, calibartion_verbalizer = \
                trainer(prompt_model, train_dataloader, dev_dataloader, devset, dataset_name, tokenizer, WrapperClass)   
            plm = classifier.plm ### this line is important, to record the tuned backbone model
            classifier.plm.save_pretrained(path)
            _, _, model_config, _ = load_plm(model_name.split("-")[0], MODEL_PATH[model_name])
            model_config.save_pretrained(path)
            tokenizer.save_pretrained(path)
            if method == "E-PLM": # use the original prompt_model as answer-provider
                classifier = prompt_model

        if method in ["I-PLM", "I-iterative", "I-multitask"]:
            # evaluate performance using the tuned model
            print("Performance:")
            if dataset_name not in ["sst2", "yahoo_answers_topics"]:
                acc_path = f"./results/metrics/ood/{dataset_name}/accuracy_of_{method}/{model_name}"
                os.makedirs(acc_path, exist_ok=True)
                for ood_name, test_dataloader in dataloader_dict.items(): # original dataloader
                    if ood_name == "test":
                        ood_name = f"{dataset_name}_iid"
                    acc = evaluation(test_dataloader, classifier, dataset_name, model_name, ood_name, method, seed)
                    np.save(os.path.join(acc_path, f"{ood_name}-{seed}.npy"), acc)
            else:
                for ood_name, test_dataloader in dataloader_dict.items():
                    if ood_name == "test":
                        ood_name = f"{dataset_name}_iid"
                        evaluation(test_dataloader, classifier, dataset_name, model_name, ood_name, method, seed)
                    compute_entropy(test_dataloader, classifier, dataset_name, model_name, ood_name, method, seed)

        
        # re-wrap for evaluating calibration
        for ood_name, test_dataloader in dataloader_dict.items():
            test_dataloader =  wrap_verbalized_testloader(classifier, test_dataloader, dataset[ood_name], calibartion_template, tokenizer, WrapperClass)
            dataloader_dict[ood_name] = test_dataloader

        # change the mode from performance to calibration. 
        # use the tuned plm as the backbone model to wrap the calibrater.
        print("Wrap the calibrater")
        calibrater = PromptForClassification(plm=plm, template=calibartion_template, verbalizer=calibartion_verbalizer, freeze_plm=False).cuda()

    # evaluate calibration
    print("Calibration:")
    if dataset_name not in ["sst2", "yahoo_answers_topics"]:
        for ood_name, test_dataloader in dataloader_dict.items():
            if ood_name == "test":
                ood_name = f"{dataset_name}_iid"
            evaluation(test_dataloader, calibrater, dataset_name, model_name, ood_name, method+"-calibration", seed)
    else:
        for ood_name, test_dataloader in dataloader_dict.items():
            if ood_name == "test":
                ood_name = f"{dataset_name}_iid"
                evaluation(test_dataloader, calibrater, dataset_name, model_name, ood_name, method+"-calibration", seed)
            compute_entropy(test_dataloader, calibrater, dataset_name, model_name, ood_name, method+"-calibration", seed)

    print("finish evaluation")



def main(args):
    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    method = args.method
    seed = args.seed
    # exit()
    ood_dataset_path = os.path.join(dataset_path, "ood")

    dataset = {}

    processer = PROCESSER[dataset_name]() if method != "eda" else EDA_PROCESSER[dataset_name]()
    dataset['train'] = processer.get_examples(dataset_path, "train")
    print(dataset_name)
    print(len(dataset['train']))


    plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], model_path)

    ood_model_path = f"./model_cache/ood_{dataset_name}/{model_name}/{method}/{seed}" \
                        if method in ["label_smoothing", "ensemble", "eda"] \
                        else f"./model_cache/ood_{dataset_name}/{model_name}/Vanilla/{seed}"
                        
    if os.path.exists(ood_model_path):
        print("Load plm from cache")
        model_config.save_pretrained(ood_model_path)
        tokenizer.save_pretrained(ood_model_path)
        del plm, tokenizer, model_config, WrapperClass
        plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], ood_model_path)

    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/TextClassification/{dataset_name}/manual_template.txt", choice=0)
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"scripts/TextClassification/{dataset_name}/manual_verbalizer.txt")
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()


    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")

    if not os.path.exists(ood_model_path):
        print("Train from scratch")
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)

        loss_func = torch.nn.CrossEntropyLoss() if method != "label_smoothing" else LabelSmoothingLoss(num_classes)
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
                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        prompt_model.plm.save_pretrained(ood_model_path)
        print("save model")
        print("finish training")
    

    dataset["dev"] = processer.get_examples(dataset_path, "dev")
    dev_dataloader = PromptDataLoader(dataset=dataset["dev"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
            batch_size=16,shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head")

    if method == "temperature_scaling":
        global T
        T_path = f"./results/ood/{dataset_name}/{model_name}/{method}/T.npy"
        if os.path.exists(T_path):
            T = np.load(T_path)
            print("Load T from disk")
            print(T)
        else:
            T = temperature_scaling(dev_dataloader, prompt_model)
            os.makedirs(f"./results/ood/{dataset_name}/{model_name}/{method}", exist_ok=True)
            np.save(T_path, T)
            
    eval(prompt_model, train_dataloader, dataset["dev"], dev_dataloader, processer, dataset_path, ood_dataset_path, mytemplate, tokenizer, WrapperClass, dataset_name, model_name, method, seed)
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="roberta")
    parser.add_argument('--scale', type=str, default="base")
    parser.add_argument('--dataset_name', type=str, default="amazon_food")
    parser.add_argument('--method', type=str, default="Vanilla", 
                        choices=["Vanilla", "label_smoothing", "ls","temperature_scaling", "ts", "ensemble", "eda",
                                "E-MLP", "E-PLM", "I-PLM", "I-iterative", "I-multitask"])
    args = parser.parse_args()
    device = torch.device("cuda")
    
    if args.dataset_name == "yahoo":
        args.dataset_name = "yahoo_answers_topics"
    if args.method == "ls":
        args.method = "label_smoothing"
    if args.method == "ts":
        args.method = "temperature_scaling"

    args.model_path = MODEL_PATH[args.model_name]
    args.dataset_path = DATASET_PATH[args.dataset_name]
    args.num_classes = NUM_CLASSES[args.dataset_name]

    T = 0
    if args.method == "ensemble":
        args.repeats = 5

    for i in range(args.repeats):
        set_seed(i)
        args.seed = i
        main(args)
    
