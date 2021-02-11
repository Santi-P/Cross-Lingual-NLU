import torch
import numpy as np

# Dataloading and Batching classes
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

# Training classes
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification

# Tokenizers and Models
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaTokenizer,
    XLMRobertaTokenizer,
)

# joint XLM-R NL
from XLMTransformers import *
from typing import Dict, NamedTuple, Optional
from sklearn.metrics import classification_report, f1_score, accuracy_score


from conll_loader import ConLLLoader, intent_labels_list, slot_labels_list
from sklearn_crfsuite import metrics as seq_metrics


# Obsolete. Now uses the recommended solution using list comprehensions.
# def remove_padding(gold_slots, pred_slots):

#     zipped = zip(gold_slots, pred_slots)
#     sanitized_gold = []
#     sanitized_pred = []
#     for gold, pred in zipped:
#         if gold != -100:
#             sanitized_gold.append(gold)
#             sanitized_pred.append(pred)
#     return sanitized_gold, sanitized_pred


def show_align_labels(p,tokenized_utterances, intent_label_list, slot_label_list, ):
    intent_predictions, slot_predictions = p.predictions
    intent_labels, slot_labels = p.label_ids

    slot_predictions = np.argmax(slot_predictions, axis=2)
    intent_predictions = np.argmax(intent_predictions, axis=1)

    slot_predictions_clean = [
        [slot_label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions, slot_labels)
    ]
    slot_labels_clean = [
        [slot_label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions, slot_labels)
    ]

    for pred, gold, itent_pred, intent_gold, utter in zip(slot_predictions_clean, slot_labels_clean,intent_predictions, intent_predictions,tokenized_utterances):
        print()
        print(intent_label_list[itent_pred],"\t",intent_label_list[intent_gold])

        for  tok_pred, tok_gold,real_tok in zip(pred,gold,utter):
            print(real_tok,"\t", tok_pred,"\t", tok_gold)

def joint_classification_report(p, intent_label_list, slot_label_list, verbose=True):
    intent_predictions, slot_predictions = p.predictions
    intent_labels, slot_labels = p.label_ids

    slot_predictions = np.argmax(slot_predictions, axis=2)
    intent_predictions = np.argmax(intent_predictions, axis=1)

    slot_predictions_clean = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions, slot_labels)
    ]
    slot_labels_clean = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions, slot_labels)
    ]

    labels_slot = list(range(len(slot_label_list)))
    labels_intent = list(range(len(intent_label_list)))
    seq_acc = seq_metrics.sequence_accuracy_score(
        slot_labels_clean, slot_predictions_clean
    )

    if verbose:
        print(
            classification_report(
                intent_labels,
                intent_predictions,
                target_names=intent_label_list,
                labels=labels_intent,
                digits = 4,
            )
        )
        print(
            seq_metrics.flat_classification_report(
                slot_labels_clean,
                slot_predictions_clean,
                target_names=slot_label_list,
                labels=labels_slot,
                digits = 4,
            )
        )
        print("sequence accuracy: ", seq_acc)

    # In efficient
    # can be done in one run and pretty print output reconstructed from dictionary
    slot_res_dict = seq_metrics.flat_classification_report(
        slot_labels_clean,
        slot_predictions_clean,
        target_names=slot_label_list,
        labels=labels_slot,
        output_dict=True,
        digits = 5,
    )

    intent_res_dict = classification_report(
        intent_labels,
        intent_predictions,
        target_names=intent_label_list,
        labels=labels_intent,
        output_dict=True,
        digits = 5,
    )

    return {
        "sequence_accuracy": seq_acc,
        "slot_results": slot_res_dict,
        "intent_results": intent_res_dict,
    }


def exact_match(p):

    intent_predictions, slot_predictions = p.predictions
    intent_labels, slot_labels = p.label_ids

    intent_predictions = np.argmax(intent_predictions, axis=1)
    slot_predictions = np.argmax(slot_predictions, axis=2)

    intent_matches = (intent_labels == intent_predictions)

    slot_predictions_clean = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions, slot_labels)
    ]
    slot_labels_clean = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions, slot_labels)
    ]

    # for seq_lab, seq_pred in zip(slot_labels_clean, slot_predictions_clean):
    #     print(seq_lab, seq_pred)
    seq_match = [
        True if np.array_equal(yseq_true, yseq_pred) else False
        for yseq_true, yseq_pred in zip(slot_labels_clean, slot_predictions_clean)
    ]


    exact_matches = np.logical_and(intent_matches,seq_match)
    #print(list(zip(intent_matches,seq_match)))
    num_exact = np.sum(exact_matches)
    total = len(intent_labels)
    return num_exact/ float(total)
    #print(seq_match)


def running_metrics(p):
    intent_predictions, slot_predictions = p.predictions
    intent_labels, slot_labels = p.label_ids

    slot_predictions = np.argmax(slot_predictions, axis=2)
    intent_predictions = np.argmax(intent_predictions, axis=1)

    slot_predictions_clean = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions, slot_labels)
    ]
    slot_labels_clean = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions, slot_labels)
    ]
    intent_f1 = f1_score(intent_labels, intent_predictions, average="macro")
    intent_accuracy = accuracy_score(intent_labels, intent_predictions)
    flat_acc = seq_metrics.flat_accuracy_score(
        slot_labels_clean, slot_predictions_clean
    )
    flat_f1 = seq_metrics.flat_f1_score(
        slot_labels_clean, slot_predictions_clean, average="macro"
    )
    slt_f1_weighted = seq_metrics.flat_f1_score(
        slot_labels_clean, slot_predictions_clean, average="weighted"
    )
    return {
        "flat slot accuracy": flat_acc,
        "flat slot f1": flat_f1,
        "weighted slot f1": slt_f1_weighted,
        "intent f1": intent_f1,
        "intent accuracy": intent_accuracy,
    }


# temporary predict function
# now works with integrated prediction_loop from Trainer class
# def predict(
#     model, test_dataloader, intent_labels_list=None, slot_labels_list=None, report=True
# ):
#     gold_intent = []
#     gold_slots = []
#     pred_intent = []
#     pred_slots = []
#     gold_slots_seq = []
#     pred_slots_seq = []
#     for example in test_loader:
#         input_ids = example["input_ids"]
#         attention_mask = example["attention_mask"]
#         results = model(
#             input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
#         )
#         intent = np.argmax(results["intents"].cpu().detach().numpy(), axis=1)
#         slots = np.argmax(results["slots"].cpu().detach().numpy(), axis=2)
#         slots = slots[0]
#         real_intent = example["intent_label_ids"].tolist()
#         real_slots = example["slot_labels_ids"]
#         real_slots = [i.item() for i in real_slots]
#         # print(intent,real_intent)
#         pred_intent.extend(intent)
#         pred_slots.extend(slots)
#         gold_intent.extend(real_intent)
#         gold_slots.extend(real_slots)
#         gold_slots_seq.append(real_slots)
#         pred_slots_seq.append(slots)
#         sanitized_gold, sanitized_pred = remove_padding(gold_slots, pred_slots)
#     if report:
#         # TODO catch NONE parameters
#         print(
#             classification_report(
#                 gold_intent, pred_intent, target_names=intent_labels_list
#             )
#         )
#         print(
#             classification_report(
#                 sanitized_gold,
#                 sanitized_pred,
#                 labels=list(range(len(slot_labels_list))),
#                 target_names=slot_labels_list,
#             )
#         )
#         # print(metrics.sequence_accuracy_score(gold_slots_seq, pred_slots_seq))
#     return {"intents": pred_intent, "slots": pred_slots}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path2train_en = "/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/train-en.conllu"
    path2eval_en = (
        "/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/eval-en.conllu"
    )
    path2test_en = (
        "/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/eval-en.conllu"
    )

    pretrained_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_set = ConLLLoader(
        path2train_en, tokenizer, intent_labels_list, slot_labels_list
    )
    val_set = ConLLLoader(path2eval_en, tokenizer, intent_labels_list, slot_labels_list)
    test_set = ConLLLoader(
        path2test_en, tokenizer, intent_labels_list, slot_labels_list
    )

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=10,  # total # of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",
        learning_rate=5e-5,
        save_steps=2000,
        label_names=["intent_label_ids", "slot_labels_ids"],
        evaluation_strategy="epoch",
    )

    conf = config_init(pretrained_name)
    model = JointClassifier(conf, num_intents=12, num_slots=31)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        compute_metrics=running_metrics,
    )

    res = trainer.evaluate()
    print(res)
    trainer.train()
