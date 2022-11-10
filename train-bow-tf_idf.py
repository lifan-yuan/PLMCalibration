import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble  import RandomForestClassifier
import numpy as np
import pandas as pd
from utils.finetune_data_utils import *


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


VECTORIZER = {
    "bag_of_words": CountVectorizer,
    "tf_idf": TfidfVectorizer
}



def load_data(dataset_name):
    dataset_path = DATASET_PATH[dataset_name]
    processer = PROCESSER[dataset_name]()
    dataset = {}
    dataset['train'] = processer.get_examples(dataset_path, "train")
    dataset['test'] = processer.get_examples(dataset_path, "test")
    return dataset

def preprocess(dataset):
    train_data_features = [" ".join([text_a, text_b]) for text_a, text_b, _ in dataset['train']]
    train_labels = [l for _, _, l in dataset["train"]]

    test_data_features = [" ".join([text_a, text_b]) for text_a, text_b, _ in dataset['test']]
    test_labels = [l for _, _, l in dataset["test"]]
    
    return train_data_features, train_labels, test_data_features, test_labels

def vectorize(vectorizer, train_data_features, test_data_features):
    train_data_features = vectorizer.fit_transform(train_data_features)
    train_data_features = pd.DataFrame(train_data_features.toarray())

    test_data_features = vectorizer.transform(test_data_features)
    test_data_features = test_data_features.toarray()

    return train_data_features, test_data_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="bag_of_words", choices=["bag_of_words", "tf_idf"])
    parser.add_argument('--dataset_name', type=str, default="sst2")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name

    # load data
    dataset = load_data(dataset_name)

    # preprocess
    train_data_features, train_labels, test_data_features, test_labels = preprocess(dataset)

    # vectorize
    vectorizer = VECTORIZER[model_name](analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    train_data_features, test_data_features = vectorize(vectorizer, train_data_features, test_data_features)

    # train
    forest = RandomForestClassifier(n_estimators = 100) 
    forest = forest.fit( train_data_features, train_labels )

    # test
    allprobs = forest.predict_proba(test_data_features)

    allprobs = [max(prob) for prob in allprobs]
    allpreds = forest.predict(test_data_features)
    alllabels = test_labels

    # save
    os.makedirs(f"./results/pretrain/{dataset_name}/{model_name}", exist_ok=True)
    np.save(f"./results/pretrain/{dataset_name}/{model_name}/alllabels.npy", alllabels)
    np.save(f"./results/pretrain/{dataset_name}/{model_name}/allprobs.npy", allprobs)
    np.save(f"./results/pretrain/{dataset_name}/{model_name}/allpreds.npy", allpreds)

    print("done")