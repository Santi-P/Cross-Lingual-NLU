# training script and commandline interface for Joint NLU. 
# last edited: 10.2.2021
# SP


import argparse
import os
import sys
import warnings
import logging

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import numpy as np
from sklearn_crfsuite import metrics

from transformers import logging as trans_log
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    ProgressCallback,
)

from transformers.integrations import TensorBoardCallback, WandbCallback
from transformers.trainer_callback import PrinterCallback

from XLMTransformers import *
from sklearn.metrics import classification_report
from conll_loader import ConLLLoader, intent_labels_list, slot_labels_list

from joint_metrics import running_metrics, joint_classification_report, exact_match

# configure loggers
trans_log.set_verbosity_warning()
logging.basicConfig( level=logging.INFO)

# disable huggingface warning and wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

#suppress annoying warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")



# disable wandb for memory efficiency.
#  this does not seem to work.
# will make issue on huggingface github


# detect and enable CUDA backend
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("using CUDA backend")
else:
    logging.info("using CPU backend")
    device = torch.device("cpu")

# set seeds for reproducibility
torch.manual_seed(136)
np.random.seed(136)


parser = argparse.ArgumentParser(description="run cross lingual NLU experiment")

# train-test arguments

parser.add_argument(
    "--train_file",
    metavar="train",
    type=str,
    help="CONLLU training file",
    required=True,
)
parser.add_argument(
    "--cross_train_file",
    type=str,
    help="CONLLU cross training file. Will either be concataned to sequentially trained depending on --sequential_cross_train flag",
)

parser.add_argument(
    "--test_file",
    type=str,
    metavar="test",
    required=True,
    help="CONNLU test file to run after training",
)

parser.add_argument(
    "--cross_test_file",
    type=str,
    metavar="xtest",
    help="CONNLU test file to run after training",
)
parser.add_argument(
    "--eval_file",
    metavar="eval",
    help="CONLLU evaluation file for evaluation between epochs",
)

parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="number of training epochs. default is 10",
)
parser.add_argument(
    "--training_bs", type=int, default=32, help="training mini batch size. "
)
parser.add_argument(
    "--lr", type=float, default=5e-5, help="initial learning rate for adam optimizer."
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./models",
    help="save directory for the final model",
)
parser.add_argument("--chkpt_dir", type=str)
parser.add_argument(
    "--no_fp16",
    action = 'store_true',
    help="enable mixed-precision training. Use False if nvidia apex is unavailable",
)
parser.add_argument(
    "--run_name", type=str, default="test_model", help="name of the experiment"
)
parser.add_argument(
    "--sequential_train",
    action = 'store_true',
    help="""sequential train performs cross lingual training sequentially.
            Otherwise the second dataset will be concatenated and shuffled""",
)


args = parser.parse_args()
arg_dict = vars(args)


# refactor to use args.xxxx instead
train_path, test_path, eval_path, save_dir = (
    arg_dict["train_file"],
    arg_dict["test_file"],
    arg_dict["eval_file"],
    arg_dict["save_dir"],
)


# Check if paths are valid
# todo check for all provided paths


if not os.path.isfile(train_path):
    print("The training file does not exist")
    sys.exit()

if not os.path.isfile(test_path):
    print("The test file specified does not exist")
    sys.exit()

if eval_path != None:
    if not os.path.isfile(eval_path):
        print("The eval file  does not exist")
        sys.exit()

if save_dir != None:
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

# train
path2train_en = train_path
# test
path2test_en = test_path

#cross train
path2xtrain_en = args.cross_train_file
#cross test
path2xtest_en = args.cross_test_file

#eval...
path2eval_en = eval_path

# name of transformer model to used. Only tested on XLM-Roberta base but theoretically extensible to other models as long as they share similar dimensions.
# TODO add as argument. but make sure it works with joint model.
pretrained_name = "xlm-roberta-base"

# BUG: Sometimes Autotokenizer cannot find pre-trained models in cache and re-downloads.
# There is no problem with this except that
# Issue number #4197 on Huggingface github


# Load default tokenizer for XLM-R
tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
# default data collator for sequential data
data_collator = DataCollatorForTokenClassification(tokenizer)

# parse, tokenize data. CONLLLoader is a sub-class of torch.Dataset and shares most of the functionalities.
train_set = ConLLLoader(path2train_en, tokenizer, intent_labels_list, slot_labels_list)
test_set = ConLLLoader(path2test_en, tokenizer, intent_labels_list, slot_labels_list)

#uncomment here to perform sequential training
# Temp fix for bug
#args.sequential_train = True

# check if cross train set provided else none
if path2xtrain_en != None:
    # detect if second training set is provided
    logging.info("using cross-training")
    extrain_set = ConLLLoader(
        path2xtrain_en, tokenizer, intent_labels_list, slot_labels_list
    )
    
    # if not sequential mix datasets
    if args.sequential_train != True:
        logging.info("using combined data strategy")
        train_set = ConcatDataset([train_set, extrain_set])
        
else:
    extrain_set = None
    
    


# check if cross train provided
# else return none
if path2xtest_en != None:
    # detect if second training set is provided
    logging.info("using cross-testing")
    extest_set = ConLLLoader(
        path2xtest_en, tokenizer, intent_labels_list, slot_labels_list
    )   
else:
    #print("no x test provided")
    extest_set = None


if path2eval_en != None:
    # detect if eval dataset is provided
    eval_set = ConLLLoader(
        path2eval_en, tokenizer, intent_labels_list, slot_labels_list
    )
else:
    eval_set = None


#print(args.no_fp16)
if args.no_fp16 == True:
    fp16 = False
    logging.info("mixed precision training disabled")
else:
    fp16 = True

training_args = TrainingArguments(
    output_dir="./results",  # output directory for check points
    num_train_epochs=args.num_epochs,  # total # of training epochs
    per_device_train_batch_size=args.training_bs,  # batch size per device during training
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",
    learning_rate=args.lr,
    save_steps=5000,
    fp16=fp16,
    label_names=["intent_label_ids", "slot_labels_ids"],
    evaluation_strategy="epoch",
    run_name=args.run_name,
    save_total_limit = 3, # prevent save files from crashing computer
    #load_best_model_at_end = True, 
)

# default config for XLM-R
conf = config_init(pretrained_name)

# Initialize joint classifier model
model = JointClassifier(conf, num_intents=12, num_slots=31)


# test freeze bert
# make arg in file
#for param in model.roberta.parameters():
    #param.requires_grad = False

# setup trainer for 1st train 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=data_collator,
    compute_metrics=running_metrics,
    #callbacks = [PrinterCallback]
)

# attempt to remove overly verbose output from Trainer API
trainer.remove_callback(PrinterCallback)
trainer.remove_callback(WandbCallback)
trainer.remove_callback(TensorBoardCallback)

logging.info("starting training session")

# train on first file
trainer.train()

# train model using transformers.Trainer API
# train on cross lingual training data if sequential training strategy is used


# detect if sequential training. 
if extrain_set != None and (args.sequential_train == True):
    logging.info("performing sequential training")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=extrain_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        compute_metrics=running_metrics,
        #callbacks = [PrinterCallback]
    )
    
    trainer.train()



    
# save model
trainer.save_model(args.save_dir)




# standard p,r, f1 classification report for both intents and slots
print("trained on: ", train_path)

# predict on 1st test file
print("*"*15, "test results for ",args.test_file, "*"*15)
preds = trainer.predict(test_set)
result_dict = joint_classification_report(preds, intent_labels_list, slot_labels_list)
print("exact matches", exact_match(preds))



# predict on 2nd test file
if extest_set != None:
    print("*"*15, "cross test results for ",args.cross_test_file, "*"*15)
    preds = trainer.predict(extest_set)
    result_dict = joint_classification_report(preds, intent_labels_list, slot_labels_list)
    print("exact matches", exact_match(preds))



