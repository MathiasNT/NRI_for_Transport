#!/bin/sh
#BSUB -J mlp_tau_test_full
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -N
#BSUB -o ../logs/%J_Output_mlp_tau_test_full.out
#BSUB -e ../logs/%J_Error_mlp_tau_test_full.err

EXP_NAME=mlp_tau_test_full
mkdir "../models/${EXP_NAME}"


EPOCHS=300
KL_CYC=50
CUDA_DEVICE=0
BATCH_SIZE=250
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.1


ENC_N_HID=128
DEC_N_HID=16
DEC_MSG_HID=8
DEC_GRU_HID=8

PICKUP_DATA_PATH=full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv


GUMBEL_TAU=0.1
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/MLP_hard_${GUMBEL_TAU}tau \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type mlp \
                        --loss_type nll \
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


GUMBEL_TAU=0.5
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/MLP_hard_${GUMBEL_TAU}tau \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type mlp \
                        --loss_type nll \
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

GUMBEL_TAU=0.9
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/MLP_hard_${GUMBEL_TAU}tau \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type mlp \
                        --loss_type nll \
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

GUMBEL_TAU=0.1
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/MLP_soft_${GUMBEL_TAU}tau \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type mlp \
                        --loss_type nll \
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


GUMBEL_TAU=0.5
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/MLP_soft_${GUMBEL_TAU}tau \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type mlp \
                        --loss_type nll \
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

GUMBEL_TAU=0.9
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/MLP_soft_${GUMBEL_TAU}tau \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type mlp \
                        --loss_type nll \
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

