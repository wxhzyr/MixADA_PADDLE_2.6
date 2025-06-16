export CUDA_VISIBLE_DEVICES=0
export PADDLE_HOME=/home/gaojingkai/.paddle

DATA_PATH=./sst-2
PWWS_DATA_PATH=./sst-2/pwws-rbt
TF_DATA_PATH=./sst-2/textfooler-rbt
# DATA_PATH=/home/sichenglei/contrast_imdb_ori/textfooler
# DATA_PATH=/home/sichenglei/sst-2
MODEL_PATH=roberta-base
# ALPHA=0.4
EPOCHS=5

OUTPUT_PATH=rbt-sst-tmixada-pwws-iterative
SEQLEN=128

# mix_option: 0: no mix, 1: TMix, 2: SimMix



echo "Running SimMix with TMix and Textfooler on SST-2 dataset"
## BERT-TMixADA-PWWS
python run_simMix.py \
--model_type roberta \
--mix_type tmix \
--iterative \
--attacker pwws \
--num_adv 300 \
--task_name sst-2 \
--data_dir ${DATA_PATH} \
--model_name_or_path ${MODEL_PATH} \
--output_dir ${OUTPUT_PATH} \
--max_seq_length $SEQLEN \
--mix_layers_set 7 9 12 \
--alpha 2.0 \
--num_labels 2 \
--do_lower_case \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--learning_rate 3e-5 \
--weight_decay 0.0 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--num_train_epochs $EPOCHS \
--warmup_steps 0 \
--logging_steps 200 \
--eval_all_checkpoints \
--seed 2020 \
--overwrite_output_dir \
--overwrite_cache \
--do_train \
--do_eval \
# --fp16
# --second_data_dir $SECOND_DATA_PATH \
# --third_data_dir $THIRD_DATA_PATH \


echo "Evaluating the model with PWWS attacker"
python attackEval.py  \
    --model_name_or_path ${OUTPUT_PATH}/best-ep1  \
    --model_type roberta \
    --attacker pwws \
    --data_dir ${DATA_PATH}/test.tsv \
    --max_seq_len $SEQLEN \
    --save_dir ./results/1.log

## BERT-TMixADA-Textfooler
# python run_simMix.py \
# --model_type roberta \
# --mix_type tmix \
# --iterative \
# --attacker textfooler \
# --num_adv 1500 \
# --task_name sst-2 \
# --data_dir ${DATA_PATH} \
# --model_name_or_path ${MODEL_PATH} \
# --output_dir /data/gaojingkai/clsi/rbt-sst-tmixada-textfooler-iterative \
# --max_seq_length $SEQLEN \
# --mix_layers_set 7 9 12 \
# --alpha 0.4 \
# --num_labels 2 \
# --do_lower_case \
# --per_gpu_train_batch_size 32 \
# --per_gpu_eval_batch_size 128 \
# --gradient_accumulation_steps 1 \
# --learning_rate 3e-5 \
# --weight_decay 0.0 \
# --adam_epsilon 1e-8 \
# --max_grad_norm 1.0 \
# --num_train_epochs $EPOCHS \
# --warmup_steps 0 \
# --logging_steps 200 \
# --eval_all_checkpoints \
# --seed 2020 \
# --overwrite_output_dir \
# --overwrite_cache \
# --do_train \
# --fp16
# # --second_data_dir $SECOND_DATA_PATH \
# # --third_data_dir $THIRD_DATA_PATH \



# python attackEval.py  \
# --model_name_or_path /data/gaojingkai/clsi/rbt-sst-tmixada-textfooler-iterative \
# --model_type roberta \
# --attacker textfooler \
# --data_dir ${DATA_PATH}/test.tsv \
# --max_seq_len $SEQLEN \
# --save_dir ./results
