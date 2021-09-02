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
#BSUB -o ../logs/%J_Output_mlp_bike.out
#BSUB -e ../logs/%J_Error_mlp_bike.err

EXP_NAME=mlp_bike_test
mkdir "../models/${EXP_NAME}"


EPOCHS=300
KL_CYC=50
CUDA_DEVICE=0
BATCH_SIZE=30
BURN_IN_STEPS=12
SPLIT_LEN=24
EDGE_RATE=0.1
GUMBEL_TAU=0.5


PICKUP_DATA_PATH=bike_data/
WEATHER_DATA_PATH=bike_data/bike_weather.csv

ENC_N_HID=32
DEC_N_HID=32
DEC_MSG_HID=32
DEC_GRU_HID=32

python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/mlp_bike\
                        --epochs ${EPOCHS} \
                        --encoder_type mlp \
                        --loss_type nll \
			--kl_cyc ${KL_CYC} \
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
			--gumbel_hard \
			--use_weather

