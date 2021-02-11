import pandas as pd
import json

class LoadData:
    def __init__(self,file_name):
        self.file_name = file_name
        self.full_data = list()
        self.data = list()
        self.data_loader()

    def data_loader(self):
        print("opening", self.file_name)
        with open(self.file_name,"r") as f:
            for line in f:
                #try:
                items = line.split("\t")
                di = json.loads(items[-1])
                #print(di['tokenizations'][0]['tokens'])

                self.full_data.append(items)
                self.data.append((items[0],di['tokenizations'][0]['tokens']))
                #except:
                    #print(line)
                    #pass
        return 0

    def data_generator(self):
        for item in self.data:
            yield item

def df_format(f_name, mapping = None):
    data = LoadData(f_name)
    parsed = [(intent,  " ".join(sentence)) for intent, sentence in data.data_generator()]
    intents, sentences = [i[0] for i in parsed], [j[1] for j in parsed]
    #print(sentences)
    slen = max([len(sen) for sen in sentences])
    if mapping == None:
        mapping = dict(enumerate(set(intents)))

    reverse_map = {intent: ind for ind, intent in mapping.items()}

    record = [(sent, reverse_map[intent]) for intent, sent in parsed]
    df = pd.DataFrame.from_records(record,columns = ["text", "labels"])

    return df, mapping







if __name__ == "__main__":
    pass

    en_df, en_mapping = df_format(("/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/train-en.tsv"))
    print(en_df)
    en_df.to_pickle("en_train.p")
    print(en_mapping)
    #
    # en_df_eval, en_mapping = df_format("/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/en/train-en.tsv",en_mapping)
    # en_df_eval.to_pickle("en_eval.p")
    #
    # es_df, es_mapping = df_format(("/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/es/train-es.tsv"))
    # es_df.to_pickle("es_train.p")
    # print(es_mapping)
    #
    # es_df_eval, es_mapping = df_format("/home/santi/BA/multilingual_task_oriented_dialog_slotfilling/es/eval-es.tsv",es_mapping)
    #
    # es_df_eval.to_pickle("es_eval.p")

