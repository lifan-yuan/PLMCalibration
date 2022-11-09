import argparse
import numpy as np
import math
import os



def ece_equal_mass(allprobs, allpreds, alllabels):

    probs_of_bins = {} 
    preds_of_bins = {}
    labels_of_bins = {}


    for bin in range(100):
        probs_of_bins[bin] = []
        preds_of_bins[bin] = []
        labels_of_bins[bin] = []
            
    # sort by prob
    data = zip(allprobs, allpreds, alllabels)
    data = sorted(data)
    allprobs, allpreds, alllabels = zip(*data)
    bin_num = 100
    num_samples_per_bin = math.ceil(len(alllabels) / bin_num)
    for i, (prob, pred, label) in enumerate(zip(allprobs, allpreds, alllabels)):
            bin = int(i / num_samples_per_bin)
            probs_of_bins[bin].append(prob)
            preds_of_bins[bin].append(pred)
            labels_of_bins[bin].append(label)

    ECE = 0
    for bin in range(100):
        probs = probs_of_bins[bin]
        preds = preds_of_bins[bin]
        labels = labels_of_bins[bin]
        avg_probs = sum([prob for prob in probs]) /len(probs) if len(probs) != 0 else 0
        bin_acc = sum([int(i==j) for i,j in zip(preds, labels)]) / len(probs) if len(probs) != 0 else 0
        ECE += abs(bin_acc-avg_probs) * len(probs)
    
    return ECE / len(allprobs)


def ece_equal_interval(allprobs, allpreds, alllabels):

    probs_of_bins = {}
    preds_of_bins = {}
    labels_of_bins = {}


    for bin in range(1, 11):
        probs_of_bins[bin] = []
        preds_of_bins[bin] = []
        labels_of_bins[bin] = []
            
    for prob, pred, label in zip(allprobs, allpreds, alllabels):
        bin = math.ceil(prob * 10)
        probs_of_bins[bin].append(prob)
        preds_of_bins[bin].append(pred)
        labels_of_bins[bin].append(label)
 
    ECE = 0
    for bin in range(1,11):
        probs = probs_of_bins[bin]
        preds = preds_of_bins[bin]
        labels = labels_of_bins[bin]
        avg_probs = sum([prob for prob in probs]) /len(probs) if len(probs) != 0 else 0
        bin_acc = sum([int(i==j) for i,j in zip(preds, labels)]) / len(probs) if len(probs) != 0 else 0
        ECE += abs(bin_acc-avg_probs) * len(probs)
    
    return ECE / len(allprobs)



def compute_ece(allprobs_list, allpreds_list, alllabels_list):

    if not isinstance(allprobs_list[0], list):
        allprobs_list = [allprobs_list]
        allpreds_list = [allpreds_list]
        alllabels_list = [alllabels_list]
    
    acc_list = []
    avg_prob_list = []
    
    ECE_equal_mass_list = []
    ECE_equal_mass_subset_list = {0:[], 1:[]}

    ECE_equal_interval_list = []
    ECE_equal_interval_subset_list = {0:[], 1:[]}


    for allprobs, allpreds, alllabels in zip(allprobs_list, allpreds_list, alllabels_list):

        avg_prob = np.mean(allprobs)
        avg_prob_list.append(avg_prob)
        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)]) / len(alllabels)
        acc_list.append(acc)

        ECE_equal_mass_subset_list[0].append(np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred != label]))
        ECE_equal_mass_subset_list[1].append(1-np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred == label]))
        ECE_equal_mass = ece_equal_mass(allprobs, allpreds, alllabels)
        ECE_equal_mass_list.append(ECE_equal_mass)

        ECE_equal_interval_subset_list[0].append(np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred != label]))
        ECE_equal_interval_subset_list[1].append(1-np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred == label]))
        ECE_equal_interval = ece_equal_interval(allprobs, allpreds, alllabels)
        ECE_equal_interval_list.append(ECE_equal_interval)


    avg_ECE_equal_mass_subset = {0: np.mean(ECE_equal_mass_subset_list[0]),1: np.mean(ECE_equal_mass_subset_list[1])}
    std_ECE_equal_mass_subset = {0: np.std(ECE_equal_mass_subset_list[0]),1: np.std(ECE_equal_mass_subset_list[1])}
    avg_ECE_equal_mass = np.mean(ECE_equal_mass_list)
    std_ECE_equal_mass = np.std(ECE_equal_mass_list)

    avg_ECE_equal_interval_subset = {0: np.mean(ECE_equal_interval_subset_list[0]),1: np.mean(ECE_equal_interval_subset_list[1])}
    std_ECE_equal_interval_subset = {0: np.std(ECE_equal_interval_subset_list[0]),1: np.std(ECE_equal_interval_subset_list[1])}
    avg_ECE_equal_interval = np.mean(ECE_equal_interval_list)
    std_ECE_equal_interval = np.std(ECE_equal_interval_list)


    avg_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    avg_probs = np.mean(avg_prob_list)
    std_probs = np.std(avg_prob_list)

    print("acc:", avg_acc, std_acc)
    print("avg_probs:", avg_probs, std_probs)
    print("|avg_acc-avg_prob| =", abs(avg_acc - avg_probs))

    # ece_equal_mass
    for key in range(2):
        print(f"ECE_equal_mass on subsets [{key}]:", avg_ECE_equal_mass_subset, std_ECE_equal_mass_subset)
    print("ECE_equal_mass: ", avg_ECE_equal_mass, std_ECE_equal_mass)

    # ece_equal_interval
    for key in range(2):
        print(f"ECE_equal_interval on subsets [{key}]:", avg_ECE_equal_interval_subset, std_ECE_equal_interval_subset)
    print("ECE_equal_interval: ", avg_ECE_equal_interval, std_ECE_equal_interval)


    print()

    
    with open(f"./metrics/{setting}/{dataset_name}/{model_name}.tsv", "a") as f:
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                                                    avg_acc, avg_probs, abs(avg_acc - avg_probs),
                                                    avg_ECE_equal_mass, avg_ECE_equal_mass_subset[1], avg_ECE_equal_mass_subset[0],
                                                    avg_ECE_equal_interval, avg_ECE_equal_interval_subset[1], avg_ECE_equal_interval_subset[0],
                                                    std_acc, std_probs, abs(std_acc - std_probs),
                                                    std_ECE_equal_mass, std_ECE_equal_mass_subset[1], std_ECE_equal_mass_subset[0],
                                                    std_ECE_equal_interval, std_ECE_equal_interval_subset[1], std_ECE_equal_interval_subset[0]
                                                ), file=f)



def shots():
    for shot in [2**i for i in range(25)]:
        if not os.path.exists(f"./results/shots/{dataset_name}/{model_name}/{shot}-shots"):
            continue
        allprobs_list = []
        allpreds_list = []
        alllabels_list = []
        for seed in range(5):
            if not os.path.exists(f"./results/shots/{dataset_name}/{model_name}/{shot}-shots/{seed}"):
                continue
            allprobs = np.load(f"./results/shots/{dataset_name}/{model_name}/{shot}-shots/{seed}/allprobs.npy").tolist()
            allpreds = np.load(f"./results/shots/{dataset_name}/{model_name}/{shot}-shots/{seed}/allpreds.npy").tolist()
            alllabels = np.load(f"./results/shots/{dataset_name}/{model_name}/{shot}-shots/{seed}/alllabels.npy").tolist()
            allprobs_list.append(allprobs)
            allpreds_list.append(allpreds)
            alllabels_list.append(alllabels)

        compute_ece(allprobs_list, allpreds_list, alllabels_list)


def dynamics():
    STEP = {
        "sst2": 100,
        "agnews": 100,
        "mnli": 5000,
        "yahoo_answers_topics": 2000
    }
    step = STEP[dataset_name]
    for step in range(0, 100000000, step):
        if not os.path.exists(f"./results/dynamics/{dataset_name}/{model_name}/{step}"):
            break
        alllabels = np.load(f"./results/dynamics/{dataset_name}/{model_name}/{step}/alllabels.npy")
        allprobs = np.load(f"./results/dynamics/{dataset_name}/{model_name}/{step}/allprobs.npy")
        allpreds = np.load(f"./results/dynamics/{dataset_name}/{model_name}/{step}/allpreds.npy")
        compute_ece(allprobs, allpreds, alllabels)


def delta_adapter():
    seed = 0
    for bottleneck_dim in [1, 4, 16, 64, 256, 1024]:
        if not os.path.exists(f"./results/delta-adapter/{dataset_name}/{model_name}/{bottleneck_dim}-dim/{seed}"):
            continue
        #print(f"at {bottleneck_dim} dim:")
        alllabels = np.load(f"./results/delta-adapter/{dataset_name}/{model_name}/{bottleneck_dim}-dim/{seed}/alllabels.npy")
        allprobs = np.load(f"./results/delta-adapter/{dataset_name}/{model_name}/{bottleneck_dim}-dim/{seed}/allprobs.npy")
        allpreds = np.load(f"./results/delta-adapter/{dataset_name}/{model_name}/{bottleneck_dim}-dim/{seed}/allpreds.npy")
        compute_ece(allprobs, allpreds, alllabels)


def delta_soft():
    seed = 0
    for soft_token_num in [1, 5, 10, 20, 50]:
        
        if not os.path.exists(f"./results/delta-soft/{dataset_name}/{model_name}/{soft_token_num}-token/{seed}"):
            continue
        #print(f"at {bottleneck_dim} dim:")
        alllabels = np.load(f"./results/delta-soft/{dataset_name}/{model_name}/{soft_token_num}-token/{seed}/alllabels.npy")
        allprobs = np.load(f"./results/delta-soft/{dataset_name}/{model_name}/{soft_token_num}-token/{seed}/allprobs.npy")
        allpreds = np.load(f"./results/delta-soft/{dataset_name}/{model_name}/{soft_token_num}-token/{seed}/allpreds.npy")
        compute_ece(allprobs, allpreds, alllabels)


def scale():
    sizes = ["tiny", "mini", "small", "medium", "base", "large"] if model_name == "bert" \
            else ["small", "base", "large", "3b"]
    for size in sizes:
        if not os.path.exists(f"./results/scale/{dataset_name}/{model_name}/{size}"):
            break
        alllabels = np.load(f"./results/scale/{dataset_name}/{model_name}/{size}/alllabels.npy")
        allprobs = np.load(f"./results/scale/{dataset_name}/{model_name}/{size}/allprobs.npy")
        allpreds = np.load(f"./results/scale/{dataset_name}/{model_name}/{size}/allpreds.npy")
        compute_ece(allprobs, allpreds, alllabels)


def pretrain():
    
    for model in [f"{model_name}-pretrain", f"{model_name}-random", "lstm", "tf_idf", "bag_of_words"]:
        allprobs = np.load(f"./results/pretrain/{dataset_name}/{model}/allprobs.npy").tolist()
        allpreds = np.load(f"./results/pretrain/{dataset_name}/{model}/allpreds.npy").tolist()
        alllabels = np.load(f"./results/pretrain/{dataset_name}/{model}/alllabels.npy").tolist()
    
        compute_ece(allprobs, allpreds, alllabels)


def ood():
    OOD_DATASET = {
        "mnli": ["mnli_iid", "snli", "hans", "anli"],
        "amazon_food": ["amazon_food_iid", "sst5", "semeval"],
        "civil_comments": ["civil_comments_iid", "hate_speech", "implicit_hate"],
        "dynasent": ["dynasent_iid", "amazon_food", "dsc"]
    }

    if model_name in ["t5-base-small", "t5-base-middle", "t5-small-middle", "t5-large-middle"]:
        method_list = ["E-MLP", "E-PLM", "I-PLM", "I-iterative", "I-multitask"]
    else:
        method_list = ["Vanilla", "temperature_scaling", "label_smoothing", "eda", "ensemble",
                        "E-MLP", "E-PLM", "I-PLM", "I-iterative", "I-multitask"]

    for ood_name in OOD_DATASET[dataset_name]:
        for method in method_list:
            if method in ["E-MLP", "E-PLM", "I-PLM", "I-iterative", "I-multitask"]:
                method_no_suffix = method
                method = method + "-calibration"
            seeds = 1 if method != "ensemble" else 5
            alllabels_list = []
            allprobs_list = []
            allpreds_list = []
            for seed in range(seeds):

                if "-calibration" in method:
                    allprobs = np.load(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allprobs.npy").tolist()
                    allpreds = np.load(f"./results/ood/{dataset_name}/{model_name}/{method_no_suffix}/{ood_name}/{seed}/allpreds.npy").tolist()
                    alllabels = np.load(f"./results/ood/{dataset_name}/{model_name}/{method_no_suffix}/{ood_name}/{seed}/alllabels.npy").tolist()
                else:
                    allprobs = np.load(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allprobs.npy").tolist()
                    allpreds = np.load(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allpreds.npy").tolist()
                    alllabels = np.load(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/alllabels.npy").tolist()

                alllabels_list.append(alllabels)
                allprobs_list.append(allprobs)
                allpreds_list.append(allpreds)

            compute_ece(allprobs_list, allpreds_list, alllabels_list)

            method = method.strip("-calibration")
            if method in ["E-MLP", "E-PLM", "I-PLM", "I-iterative", "I-multitask"]:
                if dataset_name == "dynasent":
                    continue
                acc_list = []
                for seed in range(seeds):
                    acc = np.load(f"./results/metrics/ood/{dataset_name}/accuracy_of_{method}/{model_name}/{ood_name}-{seed}.npy")
                    acc_list.append(acc.item())
                avg_acc = np.mean(acc_list)
                print("Ignore the above acc. It actually should be:", avg_acc)
                print("\n"*3)

                # re-write the acc
                with open(f"./metrics/ood/{dataset_name}/{model_name}.tsv", "r") as f:
                    data = []
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().split("\t")
                        data.append(line)
                    data[-1][0] = str(avg_acc)
                with open(f"./metrics/ood/{dataset_name}/{model_name}.tsv", "w") as f:
                    for line in data:
                        print("\t".join(line), file=f)


def entropy():
    OOD_DATASET = {
        "sst2": ["sst2_iid", "bookcorpus", "random_words"],
        "yahoo_answers_topics": ["yahoo_answers_topics_iid", "bookcorpus", "random_words"]
    }
    os.makedirs(f"./metrics/ood/{dataset_name}", exist_ok=True)
    with open(f"./metrics/ood/{dataset_name}/{model_name}.tsv", "w") as f:
        f.write("method\tavg_prob\tavg_entropy\n")
        for ood_dataset in OOD_DATASET[dataset_name]:
            for method in ["Vanilla", "temperature_scaling", "label_smoothing", "eda", "ensemble",
                            "E-MLP", "E-PLM", "I-PLM", "I-iterative", "I-multitask"]:
                if method in ["E-MLP", "E-PLM", "I-PLM", "I-iterative", "I-multitask"]:
                    method = method + "-calibration"

                seeds = 1 if method != "ensemble" else 5
                allprobs_list = []
                allentropy_list = []
                for seed in range(seeds):
                    allprobs = np.load(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_dataset}/{seed}/allprobs.npy").tolist()
                    allentropy = np.load(f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_dataset}/{seed}/allentropy.npy").tolist()

                    allprobs_list.append(allprobs)
                    allentropy_list.append(allentropy)
                avg_prob = np.mean(allprobs_list)
                avg_entropy = np.mean(allentropy_list)
                if "-calibration" in method:
                    method = method.rstrip("-calibration")
                f.write(f"{method}\t{avg_prob}\t{avg_entropy}\n")



COMPUTE = {
    "shots": shots,
    "dynamics": dynamics,
    "delta-adapter": delta_adapter,
    "delta-soft": delta_soft,
    "scale": scale,
    "pretrain": pretrain,
    "ood": ood,
    "entropy": entropy
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting_list', type=str, default=["ood"], nargs="+")
    parser.add_argument('--model_list', type=str, default=["t5", "roberta"], nargs="+")
    parser.add_argument('--dataset_list', type=str, default=["mnli", "amazon_food", "civil_comments"], nargs="+")
    args = parser.parse_args()

    setting_list = args.setting_list
    model_list = args.model_list
    dataset_list = args.dataset_list

    for setting in setting_list:
        for model_name in model_list:
            for dataset_name in dataset_list:

                if dataset_name == "yahoo":
                    dataset_name = "yahoo_answers_topics"
                if model_name == "roberta" and setting == "scale":
                    model_name = "bert"

                if setting != "entropy":
                    result_path = f"./metrics/{setting}/{dataset_name}"
                    os.makedirs(result_path, exist_ok=True)
                    with open(os.path.join(result_path, f"{model_name}.tsv"), "w") as f:
                        print("acc\tavg_probs\t|avg_acc-avg_prob|\tECE_mass\tECE_mass on True\tECE_mass on False\tECE_interval\tECE_interval on True\tECE_interval on False\tstd_acc\tstd__probs\t|std_acc-std_prob|\tstd_ECE_mass\tstd_ECE_mass on True\tstd_ECE_mass on False\tstd_ECE_interval\tstd_ECE_interval on True\tstd_ECE_interval on False", file=f)
                COMPUTE[setting]()

                
                
