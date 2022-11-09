# PLMCalibration

Code and data for paper "A Close Look into the Calibration of Pre-trained Language Models"



# Installation

```sh
pip install -r requirements.txt
```

You might also try to run the code with your own version of libraries, but this can lead to some bugs.



# Data Preparation

You need to download the datasets from Google Drive [[download]](https://drive.google.com/file/d/1738RctASgLd-vRIGxo4ytZFR3Kpb5nU0/view?usp=share_link "downlaod datasets from Google Drive"), and upload the folder (**TextClassification**) to the `./datasets` directory. Then, all the datasets used in the paper can be find in ./datasets/TextClassification .



# Experiments

## Question 1: Do PLMs learn to become calibrated in the training process?

To answer Question 1, we conduct fine-grained experiments to study the dynamic change in PLMs' calibration performance in training, including dataset difficulty, available training samples, training steps, the number of tunable parameters, model scale, and pre-training. Given that dataset difficulty is reflected by the datasets we choose, we only conduct seperate experiments of other factors to reveal their effects.


### Available training samples

Run:

```sh
python prompt-shots.py --model_name MODEL_NAME --dataset_name DATASET_NAME --repeats REPEATS --shots SHOTS
```

By default, the shot number will gradually increase until exceeding the size of the dataset, and the number of repetitions will automatically adjust according to the shot number. The results (probabilities, predictions and gold labels) will be recorded to `./results/shots`.


### Training steps

Run:

```sh
python prompt-dynamics.py --model_name MODEL_NAME --dataset_name DATASET_NAME
```

The results of every 100 steps will be recorded to `./results/dynamics`.


### Nnumber of tunable parameters

We consider two kinds of delta-tuning methods, i.e. Adapter and Soft Prompt-tuning.

For Adapter, run:

```sh
python prompt-delta.py --model_name MODEL_NAME --dataset_name DATASET_NAME --method adapter --parameter PARAMETER
```


For Soft Prompt, run:

```sh
python prompt-delta.py --model_name MODEL_NAME --dataset_name DATASET_NAME --method soft --parameter PARAMETER
```

The results will be recorded to `./results/delta-adapter` and `./results/delta-soft` respectively.


### Model scale

Run:

```sh
python prompt-scale.py --model_name MODEL_NAME --dataset_name DATASET_NAME --scale SCALE
```

The results will be recorded to `./results/scale`.



### Pre-training

We consider pre-trained PLM, random-initialized PLM, LSTM, BoW, and TF-IDF.

For PLM, run:

```sh
python prompt-pretrain.py --model_name MODEL_NAME --dataset_name DATASET_NAME --mode MODE
```

where `MODE` is *pretrain* or *random*.

For LSTM, run:

```sh
python train-lstm.py --dataset_name DATASET_NAME
```

For Bow and TF-IDF, run:

```sh
python train-bow-tf_idf.py --model_name MODEL_NAME --dataset_name DATASET_NAME
```

All the results will be recorded to `./results/pretrain`.




## Question 2: How effective are existing calibration methods?

To answer Question 2, we implement several calibation methods, including both *unlearnable* and *learnable* ones. To eplore their performance under Out-of-Distribution shift settings, we consider various kinds of OOD settings. All of the methods and OOD settings are implemented in one file, so you can simply run: 

```sh
python prompt-ood.py --model_name MODEL_NAME --dataset_name DATASET_NAME --method METHOD
```

The OOD setting and calibration methods can be changed by different values of `DATASET_NAME` and `METHOD`, respectively. All the results will be recorded to `./results/ood`.


Further, we change the size of dataset of the calibration task as well as the scale of the backbone model to explore the emergent ability of learnbale methods. Run:

```sh
python prompt-emergent.py --model_name MODEL_NAME --dataset_name DATASET_NAME --method METHOD --scale SCALE --dev_size DEV_SIZE
```

The results will be recored to `./results/ood/t5-SCALE-DEV_SIZE`, where the `SCALE` and `DEV_SIZE` are the augments that passed in the command line.


## Process Results

So far, we have obtained the results of probabilities, predictions and gold labels. Next, we will use the results to compute the metrics for calibration. Run:

```sh
python metric.py --setting_list SETTING_LIST --model_list MODEL_LIST --dataset_list DATASET_LIST
```

By passing `SETTING_LIST`, `MODEL_LIST` and `DATASET_LIST`, you can find the final metrics for all the experiments in the directory `./metrics`.
