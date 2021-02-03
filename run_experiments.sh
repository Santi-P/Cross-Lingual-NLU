# train english test en

python train_eval.py  \
--train_file data/en/train-en.conllu  \
--test_file data/en/test-en.conllu \
--eval_file data/en/eval-en.conllu \
--num_epochs 20 \
--save_dir models/en_train \
--run_name "en_train_en_test"\


# train th test th
python train_eval.py  \
--train_file data/th/train-th_TH.conllu  \
--test_file data/th/test-th_TH.conllu \
--eval_file data/th/eval-th_TH.conllu \
--num_epochs 20 \
--save_dir models/th_train \
--run_name "th_train_th_test"\

# train es test es
python train_eval.py  \
--train_file data/es/train-es.conllu  \
--test_file data/es/test-es.conllu \
--eval_file data/es/eval-es.conllu \
--num_epochs 20 \
--save_dir models/es_train \
--run_name "es_train_es_test"\


# xtrain en es, test es,en 

python train_eval.py  \
--train_file data/es/train-es.conllu  \
--cross_train_file data/en/train-en.conllu \
--test_file data/es/test-es.conllu \
--cross_test_file data/en/test-en.conllu \
--eval_file data/es/eval-es.conllu \
--num_epochs 20 \
--save_dir models/en_es_train \
--run_name "en_es_train"\


# xtrain en th test th en
python train_eval.py  \
--train_file data/th/train-th_TH.conllu  \
--cross_train_file data/en/train-en.conllu \
--test_file data/th/test-th_TH.conllu \
--cross_test_file data/en/test-en.conllu \
--eval_file data/th/eval-th_TH.conllu \
--num_epochs 10 \
--save_dir models/en_th_train \
--run_name "en_th_train"\




# xtrain en es, test es, sequential

# python train_eval.py  \
# --train_file data/es/train-es.conllu  \
# --cross_train_file data/en/train-en.conllu \
# --test_file data/es/test-es.conllu \
# --eval_file data/es/eval-es.conllu \
# --num_epochs 20 \
# --save_dir models/en_es_train \
# --run_name "spanish english  train and spanish test, sequential"\
# --sequential_train 


# python train_eval.py  \
# --train_file data/en/train-en.conllu  \
# --cross_train_file data/es/train-es.conllu \
# --test_file data/es/test-es.conllu \
# --cross_test_file data/en/test-en.conllu \
# --eval_file data/es/eval-es.conllu \
# --num_epochs 20 \
# --save_dir models/en_es_train_bla \
# --run_name "english spanish train and spanish test"\
