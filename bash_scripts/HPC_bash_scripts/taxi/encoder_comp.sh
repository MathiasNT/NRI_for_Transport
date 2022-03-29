#!/bin/sh
#BSUB -J encoder_taxi
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -N
#BSUB -o ../logs/mlp_taxi/%J_encoder_test.out
#BSUB -e ../logs/mlp_taxi/%J_encoder_test.err

EXP_NAME=mlp_taxi/encoder_comp
mkdir "../models/${EXP_NAME}"

ENCODER_TYPE=mlp
LOSS_TYPE=nll

EPOCHS=1000
KL_CYC=50
CUDA_DEVICE=0
BATCH_SIZE=200
BURN_IN_STEPS=48
SPLIT_LEN=60
EDGE_RATE=0.1
GUMBEL_TAU=0.5
LR=0.0005
NLL_VARIANCE=0.005
WEIGHT_DECAY=0.05

PICKUP_DATA_PATH=taxi_data/full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=taxi_data/LGA_weather_full_2019.csv
LOCAL_ADJ_PATH=taxi_data/full_manhattan/full_year_full_manhattan_local_adj.npy
DTW_ADJ_PATH=taxi_data/full_manhattan/train_full_manhattan_dtw_adj_bin.npy


DEC_N_HID=32
DEC_MSG_HID=32
DEC_GRU_HID=32


SEED=44

NLL_VARIANCE=0.0005




ENC_N_HID=64
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/long_soft_low_lr_enc${ENC_N_HID}_s${SEED} \
                        --epochs ${EPOCHS} \
                        --encoder_type ${ENCODER_TYPE} \
                        --loss_type ${LOSS_TYPE} \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --enc_n_hid ${ENC_N_HID} \
			--dec_n_hid ${DEC_N_HID} \
			--dec_msg_hid ${DEC_MSG_HID} \
			--dec_gru_hid ${DEC_GRU_HID} \
			--gumbel_tau ${GUMBEL_TAU} \
			--weight_decay ${WEIGHT_DECAY} \
			--nll_variance ${NLL_VARIANCE} \
			--prior_adj_path ${DTW_ADJ_PATH} \
			--lr ${LR} \
			--use_seed ${SEED}
			
