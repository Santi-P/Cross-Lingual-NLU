import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaModel,
    Trainer,
    TrainingArguments,
    XLMRobertaModel,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaClassificationHead,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers import PreTrainedModel
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from util import *


class IntentClassifier(nn.Module):
    "classifier head for intent detection"

    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.01):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(input_dim, input_dim * 2)
        self.activation = nn.Tanh()
        self.linear = nn.Linear(input_dim * 2, num_intent_labels)

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear(x)

class SimpleIntentClassifier(nn.Module):
    "vanilla dense layer for ablation study"
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.01):
        super(SimpleIntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(input_dim, input_dim * 2)
        self.activation = nn.Tanh()
        self.linear = nn.Linear(input_dim * 2, num_intent_labels)

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear(x)



class CustomClassificationHead(nn.Module):
    "deprecated XLM-R intent classification head"

    def __init__(self, num_labels, dropout=0.01, hidden_size=768):
        super(CustomClassificationHead, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SlotClassifier(nn.Module):
    "token classification layer"

    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.01):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class JointClassifier(RobertaPreTrainedModel):
    "XLM-R Joint Classifier"

    def __init__(self, config, num_intents=12, num_slots=31, return_dict=True):
        # bug 
        # num intents not passing through when loading model
        super(JointClassifier, self).__init__(config)

        self.num_labels = config.num_labels

        self.num_intent_labels = num_intents
        self.num_slot_labels = num_slots

        self.roberta = XLMRobertaModel(config, add_pooling_layer=True)
        self.intent_clf = IntentClassifier(768, self.num_intent_labels)
        self.slot_clf = SlotClassifier(768, self.num_slot_labels)

        self.return_dict = return_dict

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        intent_label_ids=None,
        slot_labels_ids=None,
    ):

        # pass inputs and attention masks into XLM Roberta
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # Hidden Layer output
        sequence_output = outputs[0]

        # pooler output
        cls_output = outputs[1]
        intent_logits = self.intent_clf(cls_output)
        slot_logits = self.slot_clf(sequence_output)
        total_loss = 0.0

        # if label is not empty
        if slot_labels_ids is not None:

            slot_loss_fct = nn.CrossEntropyLoss()

            # apply attention masks to padding tokens
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                # calculate x-entropy
                slot_loss = slot_loss_fct(active_logits, active_labels)

            else:
                slot_loss = slot_loss_fct(
                    slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1)
                )

            # print(slot_loss)
            total_loss += slot_loss

        if intent_label_ids is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(
                intent_logits.view(-1, self.num_intent_labels),
                intent_label_ids.view(-1),
            )
            total_loss += intent_loss



        outputs = ((intent_logits, slot_logits),) + outputs[
            2:
        ]  # add hidden states and attention if they are here
        outputs = (total_loss,) + outputs

        # return as dictionary
        if self.return_dict:
            return {"loss": total_loss, "intents": intent_logits, "slots": slot_logits}

        # tuple -> loss, intent logits, slot logits
        return outputs


def config_init(model_name):
    """simple wrapper for autoconfig"""
    conf = AutoConfig.from_pretrained(model_name)
    return conf


def tokenizer_init(model_name):
    """simple wrapper for auto tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def main(data):
    pass


if __name__ == "__main__":
    main()
