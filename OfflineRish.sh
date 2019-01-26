#!/bin/sh

export SAVE_DIR=outputs/Offline/20190116
export MODEL=../bert-model
export REPORTFILE=${SAVE_DIR}/report_token.o

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

lr_rates="1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4"

for lr in $lr_rates; do
    for t in $(seq  3)
    do
        save_model=${SAVE_DIR}/lr${lr}_t${t}/
        time=$(date "+%Y-%m-%d %H:%M:%S")
        echo "epoch :${e_num}" >> $REPORTFILE
        echo "$time" >> $REPORTFILE
        
        python3 ./run_classifier.py --task_name=offline --do_train=true --do_eval=true --do_predict=true --data_dir=./project/sms/data/all --vocab_file=$MODEL/vocab.txt --bert_config_file=$MODEL/bert_config.json --init_checkpoint=$MODEL/bert_model.ckpt --max_seq_length=256 --train_batch_size=8 --learning_rate=${lr} --num_train_epochs=15 --output_dir=$save_model
          
        cat ${save_model}eval_results.txt >> $REPORTFILE
        echo "--------" >> $REPORTFILE
    
    done
done
