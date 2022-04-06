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
#BSUB -o ../logs/mlp_taxi/%J_final_fixed.out
#BSUB -e ../logs/mlp_taxi/%J_final_fixed.err

EXP_NAME=final_models/taxi/fixed_models_two
mkdir "../models/${EXP_NAME}"

ENCODER_TYPE=fixed
LOSS_TYPE=nll

EPOCHS=800
KL_CYC=50
CUDA_DEVICE=0
BATCH_SIZE=128
BURN_IN_STEPS=48
SPLIT_LEN=60
PRED_STEPS=12
EDGE_RATE=0.1
GUMBEL_TAU=0.5
LR=0.0005
NLL_VARIANCE=0.0005
WEIGHT_DECAY=0.05

NORMALIZATION=ha

PICKUP_DATA_PATH=taxi_data/full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=taxi_data/LGA_weather_full_2019.csv




FIX_TYPE=normdtw
FIX_ADJ_PATH=taxi_data/full_manhattan/full_year_full_manhattan_local_adj.npy

#FIX_TYPE=dtw
#FIX_ADJ_PATH=taxi_data/full_manhattan/train_full_manhattan_dtw_adj_bin.npy

#FIX_TYPE=full
#FIX_ADJ_PATH=taxi_data/full_manhattan/full_adj.npy

#FIX_TYPE=empty
#FIX_ADJ_PATH=taxi_data/full_manhattan/empty_adj.npy

FIX_TYPE=normdtw
FIX_ADJ_PATH=taxi_data/full_manhattan/train_full_manhattan_norm_dtw_adj_bin.npy




DEC_N_HID=32
DEC_MSG_HID=32
DEC_GRU_HID=32

ENC_N_HID=128

SEED=43

python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/${NORMALIZATION}_lr${LR}_${FIX_TYPE}_s${SEED} \
                        --epochs ${EPOCHS} \
                        --encoder_type ${ENCODER_TYPE} \
                        --loss_type ${LOSS_TYPE} \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
			--normalize ${NORMALIZATION} \
			--use_weather \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
			--pred_steps ${PRED_STEPS} \
                        --edge_rate ${EDGE_RATE} \
                        --enc_n_hid ${ENC_N_HID} \
			--dec_n_hid ${DEC_N_HID} \
			--dec_msg_hid ${DEC_MSG_HID} \
			--dec_gru_hid ${DEC_GRU_HID} \
			--gumbel_tau ${GUMBEL_TAU} \
			--weight_decay ${WEIGHT_DECAY} \
			--nll_variance ${NLL_VARIANCE} \
			--fixed_adj_matrix_path ${FIX_ADJ_PATH} \
			--lr ${LR} \
			--kl_cyc ${KL_CYC} \
			--gumbel_hard \
			--use_seed ${SEED}
			
