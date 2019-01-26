#!/bin/sh

export SAVE_DIR=news_test/
export MODEL=../bert-model
export REPORTFILE=report_token.o
export DATA_DIR=project/news/data

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

time=$(date "+%Y-%m-%d %H:%M:%S")
save_model=${SAVE_DIR}${time}/
export INIT_MODEL=project/news/model/news_v1/bert_fix_94172

python3 ./run_classifier.py --task_name=news --do_predict=true --data_dir=$DATA_DIR --vocab_file=$MODEL/vocab.txt --bert_config_file=$MODEL/bert_config.json --init_checkpoint=$INIT_MODEL/model.ckpt-6851 --max_seq_length=256 --output_dir=$save_model
  
