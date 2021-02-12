# Cross-Lingual Natural Language Understanding

This is the source code for the experiments and models in my thesis paper. The repository consists of a custom joint NLU system proposed by Chen et al(2019). The joint NLU relies on XLM-R (Conneau, 2020) as its language model. Data for training and testing comes is collected by Schuster et al (2019). 

## Installation
`pip install -r requirements.txt`

## Usage

To replicate the experiments for Joint NLU, simply run the shell script `run_experiments.sh`. Experiments for disjoint NLU are contained in the Jupyter notebook 
`Intent_detection.ipynb` and `Slot-filling.ipynb`

`train_eval.py` handles the main training loop for the joint models. 

```
usage: train_eval.py [-h] --train_file train
                     [--cross_train_file CROSS_TRAIN_FILE] --test_file test
                     [--cross_test_file xtest] [--eval_file eval]
                     [--num_epochs NUM_EPOCHS] [--training_bs TRAINING_BS]
                     [--lr LR] [--save_dir SAVE_DIR] [--chkpt_dir CHKPT_DIR]
                     [--no_fp16] [--run_name RUN_NAME] [--sequential_train]

optional arguments:
  -h, --help            show this help message and exit
  --train_file train    CONLLU training file
  --cross_train_file CROSS_TRAIN_FILE
                        CONLLU cross training file. Will either be concataned
                        to sequentially trained depending on
                        --sequential_cross_train flag
  --test_file test      CONNLU test file to run after training
  --cross_test_file xtest
                        CONNLU test file to run after training
  --eval_file eval      CONLLU evaluation file for evaluation between epochs
  --num_epochs NUM_EPOCHS
                        number of training epochs. default is 10
  --training_bs TRAINING_BS
                        training mini batch size.
  --lr LR               initial learning rate for adam optimizer.
  --save_dir SAVE_DIR   save directory for the final model
  --chkpt_dir CHKPT_DIR
  --no_fp16             enable mixed-precision training. Use False if nvidia
                        apex is unavailable
  --run_name RUN_NAME   name of the experiment
  --sequential_train    sequential train performs cross lingual training
                        sequentially. Otherwise the second dataset will be
                        concatenated and shuffled
```

`predict.py` is used to load up a pre-trained model from a specified directory and predict/evaluate a dataset. 

```
usage: predict.py [-h] --model_path MODEL_PATH --test_file TEST_FILE [--eval]

Load and Predict/Test Joint NLU

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to pre-trained joint model
  --test_file TEST_FILE
                        path to test dataset in conllu format
  --eval                print model evaluation at end
```


[link to Colab!](https://colab.research.google.com/drive/1_bt8d7IfU-q-XBY4NhejDULxDU1wsg6g?usp=sharing)


Chen, Q., Zhuo, Z., & Wang, W. (2019). BERT for Joint Intent Classification and Slot Filling. ArXiv, abs/1902.10909.

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL.

Schuster, S., Gupta, S., Shah, R., & Lewis, M. (2019). Cross-lingual Transfer Learning for Multilingual Task Oriented Dialog. ArXiv, abs/1810.13327.
