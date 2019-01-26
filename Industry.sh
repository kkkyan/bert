#!/bin/sh

export SAVE_DIR=outputs/Industry
export MODEL=../bert-model
export REPORTFILE=${SAVE_DIR}/report_token.o

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

for e_num in $(seq 1 20)
do
save_model=${SAVE_DIR}/epoch_${e_num}/
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "epoch :${e_num}" >> $REPORTFILE
echo "$time" >> $REPORTFILE

python3 ./run_classifier_multi.py --task_name=industry --do_train=true --do_predict=true --data_dir=./ --vocab_file=$MODEL/vocab.txt --bert_config_file=$MODEL/bert_config.json --init_checkpoint=$MODEL/bert_model.ckpt --max_seq_length=256 --train_batch_size=8 --learning_rate=1e-5 --num_train_epochs=${e_num} --output_dir=$save_model
  
cat ${save_model}eval_results.txt >> $REPORTFILE
echo "--------" >> $REPORTFILE

done
