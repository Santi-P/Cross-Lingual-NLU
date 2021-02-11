# dataloader for conllu training data. 
# last edited: 10.2.2021
# SP

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaTokenizer,
    XLMRobertaTokenizer,
)
import torch
from conllu import parse

slot_labels_list = [
    "B-alarm/alarm_modifier",
    "I-reminder/reference",
    "B-reminder/reminder_modifier",
    "I-reminder/todo",
    "NoLabel",
    "B-timer/attributes",
    "B-datetime",
    "B-reminder/todo",
    "B-reminder/recurring_period",
    "B-timer/noun",
    "I-weather/noun",
    "B-negation",
    "B-reminder/noun",
    "I-weather/attribute",
    "I-alarm/alarm_modifier",
    "B-weather/noun",
    "I-datetime",
    "B-weather/attribute",
    "I-reminder/recurring_period",
    "I-location",
    "B-demonstrative_reference",
    "B-location",
    "I-reminder/reminder_modifier",
    "B-reminder/reference",
    "B-weather/temperatureUnit",
    "I-reminder/noun",
    "B-news/type",
    "I-demonstrative_reference",
    "I-negation",
    "B-alarm/recurring_period",
    "I-alarm/recurring_period",
]

intent_labels_list = [
    "reminder/cancel_reminder",
    "weather/find",
    "alarm/cancel_alarm",
    "reminder/show_reminders",
    "alarm/snooze_alarm",
    "alarm/time_left_on_alarm",
    "alarm/modify_alarm",
    "weather/checkSunset",
    "weather/checkSunrise",
    "alarm/show_alarms",
    "reminder/set_reminder",
    "alarm/set_alarm",
]


class ConLLLoader(Dataset):
    """data loader and tokenizer for CONLLU data. Based on pt Dataset class. Used by Trainer object"""

    def __init__(self, data_path, tokenizer, intent_labels, slot_labels, max_length = 40):
        super(Dataset).__init__()

        label_maps = self._build_maps(intent_labels, slot_labels)
        self.slot_map, self.intent_map, self.inv_slot, self.inv_intent = label_maps
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.dataset = self._parse_examples(data_file=data_path)

    def _build_maps(self, intent_labels, slot_labels):
        """maps intent and slot labels to a number. internally builds dictionaries"""

        # enumerate labels and cast to dictionary
        slot_map = dict(enumerate(slot_labels))
        intent_map = dict(enumerate(intent_labels))

        # invert that dictionary
        inv_slot = {v: k for k, v in slot_map.items()}
        inv_intent = {v: k for k, v in intent_map.items()}

        return slot_map, intent_map, inv_slot, inv_intent

    def _parse_examples(self, data_file):
        """parse and tokenizes CONLLU training to tensors"""
        results = list()
        with open(data_file, "r") as f:
            data = f.read()
            formatted = parse(data)

            text_list = [[t["form"] for t in sentence] for sentence in formatted]
            padded_encodings = self.tokenizer(
                text_list,
                return_tensors="pt",
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length = self.max_length
            )

            for idx, sentence in enumerate(formatted):

                intent = self.inv_intent[sentence[0]["lemma"]]
                slots = [self.inv_slot[t["upos"]] for t in sentence]
                new_labels = []

                slot_dict = dict(enumerate(slots))

                prev_label = None
                curr_label = None

                for tok in padded_encodings.word_ids(batch_index=idx):
                    if tok == None:
                        # Append Nobel or -100??
                        new_labels.append(-100)
                    else:
                        new_labels.append(slot_dict[tok])

                # print(new_labels)
                # maybe and untokenized sentence
                example = {
                    "input_ids": padded_encodings["input_ids"][idx],
                    "attention_mask": padded_encodings["attention_mask"][idx],
                    "intent_label_ids": intent,
                    "slot_labels_ids": new_labels,
                }
                results.append(example)

        return results

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    pass
    # train_path = "/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/train-en.conllu"
    # model_name = "xlm-roberta-base"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # train_set = ConLLLoader(train_path, tokenizer, intent_labels_list, slot_labels_list)
    # for i in train_set:
    #     print(i)
    # print(len(train_set))
