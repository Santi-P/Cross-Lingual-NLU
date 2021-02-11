# pre-trained loader and model prediction for joint NLU. 
# last edited: 10.2.2021
# SP

# stack overflow hack to supress annoying scikitlearn warnings.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaTokenizer,
    XLMRobertaTokenizer,
    Trainer,
    TrainingArguments
)

import numpy as np
from XLMTransformers import JointClassifier
import torch
from XLMTransformers import config_init
from conll_loader import intent_labels_list, slot_labels_list, ConLLLoader
from joint_metrics import joint_classification_report, exact_match, show_align_labels
import logging
import argparse

if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("using CUDA backend")
else:
    logging.info("using CPU backend")
    device = torch.device("cpu")

# set seeds for reproducibility
torch.manual_seed(136)
np.random.seed(136)


parser = argparse.ArgumentParser(description="Load and Predict/Test Joint NLU")



parser.add_argument(
    "--model_path",
    type=str,
    help="path to pre-trained joint model",
    required=True,
)
parser.add_argument(
    "--test_file",
    type=str,
    help="path to test dataset in conllu format",
    required=True
)

parser.add_argument(
    "--eval",
    action = "store_true",
    help="print model evaluation at end",
)

args = parser.parse_args()

# configure and instantiate joint model
conf = config_init("xlm-roberta-base")
model = JointClassifier(conf, num_intents=12, num_slots=31).from_pretrained(args.model_path)
# load roberta tokenzier
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# dummy training args. 
# needed for label names for function properly
training_args = TrainingArguments(
    output_dir="./results",  # output directory for check points
    logging_dir="./logs",
    label_names=["intent_label_ids", "slot_labels_ids"],
)

# parse and build test set
testset = ConLLLoader(args.test_file, tokenizer = tokenizer, intent_labels=intent_labels_list, slot_labels=slot_labels_list)
#load trainer to use trainer.predict()
trainer = Trainer(model=model, args= training_args)

# predict testset
results = trainer.predict(testset)
# reconstruct utterances from sentencepiece encoding
reconstructed_utterances = [tokenizer.convert_ids_to_tokens(i["input_ids"],skip_special_tokens=True) for i in testset]
# print aligned labels
show_align_labels(results,reconstructed_utterances, intent_labels_list, slot_labels_list)

if args.eval==True:
    joint_classification_report(results, intent_labels_list, slot_labels_list)
    print("exact match", exact_match(results))

# pretokenized inputs can be saved
# this is slightly wasteful in terms of memory
