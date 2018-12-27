#!/bin/sh

export SAVE_DIR=output_1/
export MODEL=../bert-model
export REPORTFILE=report_token.o

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

for e_num in $(seq 1 25)
do
save_model=${SAVE_DIR}epoch_${e_num}/
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "epoch :${e_num}" >> $REPORTFILE
echo "$time" >> $REPORTFILE

python3 ./run_classifier.py --task_name=news --do_train=true --do_eval=true --do_predict=true --data_dir=para --vocab_file=$MODEL/vocab.txt --bert_config_file=$MODEL/bert_config.json --init_checkpoint=$MODEL/bert_model.ckpt --max_seq_length=256 --train_batch_size=8 --learning_rate=1e-5 --num_train_epochs=1 --output_dir=$save_model
  
cat ${save_model}eval_results.txt >> $REPORTFILE
echo "--------" >> $REPORTFILE

done
