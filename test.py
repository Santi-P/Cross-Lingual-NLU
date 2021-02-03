from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaTokenizer,
    XLMRobertaTokenizer,
    Trainer
)

import numpy as np


from XLMTransformers import JointClassifier
import torch
from trainer import config_init

from conll_loader import intent_labels_list, slot_labels_list
conf = config_init("xlm-roberta-base")

model = JointClassifier(conf, num_intents=12, num_slots=31)
model.from_pretrained("models/en_train")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")




while True: 

    in_sent = input("enter a sentence: ")

    padded_encodings = tokenizer(
    in_sent,
    return_tensors="pt",
    padding=True,
    truncation=True,
    )


    outp = model(**padded_encodings)

    intent_logit = outp['intents']
    slot_logit = outp['slots']


    slot_predictions = np.argmax(slot_logit.cpu().detach().numpy(), axis=2)
    intent_predictions = np.argmax(intent_logit.cpu().detach().numpy(), axis=1)

    print(intent_labels_list[intent_predictions[0]])
    print(" ".join([slot_labels_list[i] for i in slot_predictions[0]]))

#print(intent_labels_list[intent_predictions[0]])

# training_args = TrainingArguments(
#     output_dir="./results",  # output directory
#     num_train_epochs=args.num_epochs,  # total # of training epochs
#     per_device_train_batch_size=args.training_bs,  # batch size per device during training
#     warmup_steps=500,  # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,  # strength of weight decay
#     logging_dir="./logs",
#     learning_rate=args.lr,
#     save_steps=2000,
#     fp16=args.fp16,
#     label_names=["intent_label_ids", "slot_labels_ids"],
#     evaluation_strategy="epoch",
#     run_name = args.run_name
# )

# conf = config_init(pretrained_name)
# model = JointClassifier(conf, num_intents=12, num_slots=31)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_set,
#     eval_dataset= eval_set,
#     data_collator=data_collator,
#     compute_metrics=running_metrics,
# )
