

    model =  JointClassifier(conf)

    mapping = {}
    with open('label_map.json','r') as f:
        mapping = json.load(f)
        mapping = {int(k):v for k,v in mapping.items()}

    en_df, en_mapping = df_format(("/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/train-en.tsv"),mapping)
    en_test, en_mapping = df_format(("/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/test-en.tsv"),mapping)

    training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs', 
    learning_rate=5e-8        # directory for storing logs
    )



# #trainer.evaluate()
# intent_results = []
# slot_results = []
# gold_intent = []
# gold_slots = []


# gold_intent = []
# gold_slots = []
# pred_intent = []
# pred_slots = []




# test_loader = DataLoader(test_set, batch_size=1)
# for example in test_loader:

#     input_ids = example["input_ids"]
#     attention_mask = example["attention_mask"]
    
#     results = model(input_ids = input_ids.to(device), attention_mask=attention_mask.to(device))
    
#     #print(results)
#     #print(results["intents"],results['slots'])
#     #real_intent, real_slots = example['intent_label_ids'],['slot_labels_ids']
#     #print(results["slots"])
#     intent = np.argmax(results["intents"].cpu().detach().numpy(), axis = 1)
#     slots = np.argmax(results['slots'].cpu().detach().numpy(), axis=2)

#     slots = slots[0]
#     real_intent = example["intent_label_ids"].tolist()
#     real_slots = example["slot_labels_ids"]

#     real_slots = [i.item() for i in real_slots]
#     #print(intent,real_intent)
    
    
#     pred_intent.extend(intent)
#     pred_slots.extend(slots)
#     gold_intent.extend(real_intent)
#     gold_slots.extend(real_slots)


    #real_slots = torch.tensor(real_slots)
    #print(real_slots)
    #slots = [slt for sent in slots for slt in sent]
    #print(intent,real_intent)

    # for sent in real_slots:
    #     print(sent)    
    #len_sents = [l.count_nonzero().item() for l in attention_mask]
    # slot_results.extend(slots)
    # intent_results.extend(intent)
    # #print(real_slots)
    # gold_intent.extend(real_intent)
    # gold_slots.extend(real_slots)
    #print(slot_results)
    #print(intent_results)
    #print(intent)

#     #print(example["intent_label_ids"])
# # print(intent_results)
# # print(slot_results) 
# #print(intent_results)
# #print(slot_results)
# #print(gold_slots)
# #print(gold_slots)
# #gold_slots = [slt for sent in gold_slots for slt in sent]

# #print(slot_results)
# #print(gold_slots)