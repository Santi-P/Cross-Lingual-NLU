import pandas as pd
from transformers import AutoTokenizer, AutoModel



class XLMIntent:
    def __init__(self):
        print("Intent classifier")

        self.model = ""
        
    def train(self, data, arg_dict):
        pass

    def evaluatate(self, data, metrics):
        pass

    def load_model(self, model_file):
        pass

    




if __name__ == "__main__":
    # classifier = pipeline("text")
    # print("hi")
    roberta = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    model = AutoModel.from_pretrained(roberta)
    print(model)
    #nlu = XLMIntent()