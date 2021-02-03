from torch.utils.data import Dataset
import torch
from util import *
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import AutoConfig
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaForSequenceClassification
import numpy as np

from sklearn.metrics import accuracy_score

class IntentLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels 

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = (self.labels[idx])
        return item






if __name__ == "__main__":
    mapping = {}
    with open('label_map.json','r') as f:
        mapping = json.load(f)
        mapping = {int(k):v for k,v in mapping.items()}
        
    en_df, en_mapping = df_format(("/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/train-en.tsv"),mapping)
    en_test, en_mapping = df_format(("/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/test-en.tsv"),mapping)


    training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size= 32 ,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10, 
    )

    roberta = "xlm-roberta-base"


    config = AutoConfig.from_pretrained(
    roberta,
    num_labels=len(mapping),
    finetuning_task= "something",
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def my_metrics(something):
        score =  accuracy_score(np.argmax(something.predictions, axis = 1),something.label_ids )
        return {"acc": score}

    tokenizer = XLMRobertaTokenizer.from_pretrained(roberta)
    model = XLMRobertaForSequenceClassification.from_pretrained("/home/santi/BA/final_code/results/checkpoint-2500")


    train_encodings = tokenizer(list(en_df["text"].values), truncation=True, padding=True)
    train_labels = list(en_df["labels"].values)

    test_encodings = tokenizer(list(en_test["text"].values), truncation=True, padding=True)
    test_labels = list(en_test["labels"].values)

    eval_dat = IntentLoader(test_encodings, test_labels)


    
    intent_dat = IntentLoader(train_encodings, train_labels)
    trainer = Trainer(model=model,args = training_args, train_dataset=intent_dat, eval_dataset = eval_dat,  compute_metrics = my_metrics)    
    #results = trainer.evaluate(eval_dat)
    trainer.train()


    model = torch.load("/home/santi/BA/final_code/results/checkpoint-2500")
    while True:
        in_string = input("say something: ")
        inp = tokenizer.encode_plus(in_string,return_tensors="pt")
        print(inp)
        results = trainer.prediction_step(model, inp, False)
        print(results[1][0])
        res_idx = torch.argmax(results[1][0] )
        print(res_idx)
        print(mapping[res_idx.item()])