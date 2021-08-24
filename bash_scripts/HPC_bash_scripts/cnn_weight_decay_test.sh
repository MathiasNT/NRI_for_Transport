#!/bin/sh
#BSUB -J cnn_weight_decay_test
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -N
#BSUB -o ../logs/%J_Output_cnn_weight_decay_test.out
#BSUB -e ../logs/%J_Error_cnn_weight_decay_test.err

EXP_NAME=cnn__weight_decay_test
mkdir "../models/${EXP_NAME}"


EPOCHS=5
WEIGHT_DECAY=0.005
CUDA_DEVICE=0
BATCH_SIZE=200
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.1
ENCODER_LR_RATE=0.1
N_EDGE_TYPES=2

PICKUP_DATA_PATH=full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv


ENC_N_HID=96
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/cnn_hid${ENC_N_HID}_init_no_anneal_weight_decay${WEIGHT_DECAY} \
                        --epochs ${EPOCHS} \
                        --encoder_type cnn \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --encoder_lr_frac ${ENCODER_LR_RATE} \
                        --n_edge_types ${N_EDGE_TYPES} \
                        --enc_n_hid ${ENC_N_HID} \
			--gumbel_hard \
			--init_weights \
			--weight_decay ${WEIGHT_DECAY}
