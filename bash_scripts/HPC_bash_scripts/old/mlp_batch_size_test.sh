#!/bin/sh
#BSUB -J mlp_bs_test
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -N
#BSUB -o ../logs/%J_Output_mlp_bs_test.out
#BSUB -e ../logs/%J_Error_mlp_bs_test.err

EXP_NAME=mlp_bs_test
mkdir "../models/${EXP_NAME}"


EPOCHS=300
KL_CYC=50
CUDA_DEVICE=0
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.1
GUMBEL_TAU=0.5

ENC_N_HID=128
DEC_N_HID=16
DEC_MSG_HID=8
DEC_GRU_HID=8

PICKUP_DATA_PATH=full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv


BATCH_SIZE=50
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/mlp_bs${BATCH_SIZE} \
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


BATCH_SIZE=100
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/mlp_bs${BATCH_SIZE} \
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


BATCH_SIZE=200
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/mlp_bs${BATCH_SIZE} \
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
			

BATCH_SIZE=400
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/mlp_bs${BATCH_SIZE} \
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
