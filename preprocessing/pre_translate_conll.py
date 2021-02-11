


from google_trans_new import google_translator

translator = google_translator()


def pre_translate(path_2_conllu):
    
    
     with open(path_2_conllu, "r") as f:
            
            prev_idx = 0 
            tok_buffer = []
            
            for line in f:
                if line[0] != '#' and not line.isspace():
                      #whitespace tokenize
                    line = line.split()


                    trans_text = (translator.translate(line[1], lang_src='en',lang_tgt = "th"))
    

                    print(line[0], trans_text, "\t".join(line[2:]))
                else:
                    print(line)
if __name__ == "__main__":
    print("hi")
    
    pre_translate("data/en/train-en.conllu")
