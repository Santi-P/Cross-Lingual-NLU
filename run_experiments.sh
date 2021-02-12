
# training script for replicating experiments in the paper. 
# last edited: 10.2.2021
# SP

# train english test en
# base line eng


python train_eval.py  \
--train_file data/en/train-en.conllu  \
--test_file data/en/test-en.conllu \
--eval_file data/en/eval-en.conllu \
--num_epochs 20 \
--save_dir models/en_train \
--run_name "en_train_en_test"\


# train th test th
# baseline thai
python train_eval.py  \
--train_file data/th/train-th_TH.conllu  \
--test_file data/th/test-th_TH.conllu \
--eval_file data/th/eval-th_TH.conllu \
--num_epochs 20 \
--save_dir models/th_train \
--run_name "th_train_th_test"\

# train es test es
# baseline spanish
python train_eval.py  \
--train_file data/es/train-es.conllu  \
--test_file data/es/test-es.conllu \
--eval_file data/es/eval-es.conllu \
--num_epochs 20 \
--save_dir models/es_train \
--run_name "es_train_es_test"\


# xtrain en es, test es,en 
# cross train english spanish non sequential
python train_eval.py  \
--train_file data/es/train-es.conllu  \
--cross_train_file data/en/train-en.conllu \
--test_file data/es/test-es.conllu \
--cross_test_file data/en/test-en.conllu \
--eval_file data/es/eval-es.conllu \
--num_epochs 20 \
--save_dir models/en_es_train \
--run_name "en_es_train"\


#xtrain en th test th en 
# cross train english thai non sequential
python train_eval.py  \
--train_file data/th/train-th_TH.conllu  \
--cross_train_file data/en/train-en.conllu \
--test_file data/th/test-th_TH.conllu \
--cross_test_file data/en/test-en.conllu \
--eval_file data/th/eval-th_TH.conllu \
--num_epochs 20 \
--save_dir models/en_th_train \
--run_name "en_th_train"\

#xtrain en es, test es, sequential

python train_eval.py  \
--train_file data/en/train-en.conllu   \
--cross_train_file data/es/train-es.conllu  \
--cross_test_file data/en/test-en.conllu \
--test_file data/es/test-es.conllu \
--eval_file data/es/eval-es.conllu \
--num_epochs 20 \
--save_dir models/en_es_sequential_train \
--run_name "spanish english  train and spanish test, sequential"\
--sequential_train 

#x  train en th, test th, sequential
python train_eval.py  \
--train_file data/en/train-en.conllu  \
--cross_train_file data/th/train-th_TH.conllu \
--test_file data/en/test-en.conllu \
--cross_test_file data/th/test-th_TH.conllu \
--eval_file data/th/eval-th_TH.conllu \
--num_epochs 20 \
--save_dir models/en_th_sequential_train  \
--sequential_train \


# arg test
# python train_eval.py  \
# --train_file data/en/test-en.conllu  \
# --cross_train_file data/th/test-th_TH.conllu \
# --test_file data/en/test-en.conllu \
# --cross_test_file data/th/test-th_TH.conllu \
# --eval_file data/th/eval-th_TH.conllu \
# --num_epochs 1 \
# --save_dir models/bla \
# --sequential_train \
# --no_fp16