#!/bin/sh
#BSUB -J mlp_bike
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -N
#BSUB -o ../logs/mlp_bike/%J_Output_mlp_bike.out
#BSUB -e ../logs/mlp_bike/%J_Error_mlp_bike.err

EXP_NAME=mlp_bike_test
mkdir "../models/${EXP_NAME}"


EPOCHS=600
KL_CYC=50
CUDA_DEVICE=0
BATCH_SIZE=16
BURN_IN_STEPS=12
SPLIT_LEN=24
PRED_STEPS=12
EDGE_RATE=0.1
GUMBEL_TAU=0.5
N_EDGE_TYPES=2
LR=0.0005
WEIGHT_DECAY=0.0001
NLL_VARIANCE=0.0005


PICKUP_DATA_PATH=bike_data/
WEATHER_DATA_PATH=bike_data/bike_weather.csv
PRIOR_ADJ_PATH=bike_data/bike_adj_bin.npy

CHECK_POINT_PATH="../models/mlp_bike_test/long_mlp_lr0.0005_s42/checkpoint_model_dict.pth"

ENC_N_HID=64
DEC_N_HID=64
DEC_MSG_HID=64
DEC_GRU_HID=64

SEED=42
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/checkpointed_long_mlp_lr${LR}_s${SEED}\
                        --epochs ${EPOCHS} \
                        --encoder_type mlp \
                        --loss_type nll \
			--cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
			--weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
			--pred_steps ${PRED_STEPS} \
                        --edge_rate ${EDGE_RATE} \
			--n_edge_types ${N_EDGE_TYPES} \
                        --enc_n_hid ${ENC_N_HID} \
			--dec_n_hid ${DEC_N_HID} \
			--dec_msg_hid ${DEC_MSG_HID} \
			--dec_gru_hid ${DEC_GRU_HID} \
			--gumbel_tau ${GUMBEL_TAU} \
			--gumbel_hard \
			--weight_decay ${WEIGHT_DECAY} \
			--nll_variance ${NLL_VARIANCE} \
			--prior_adj_path ${PRIOR_ADJ_PATH} \
			--use_seed ${SEED} \
			--use_weather \
			--lr ${LR} \
			--checkpoint_path ${CHECK_POINT_PATH}
