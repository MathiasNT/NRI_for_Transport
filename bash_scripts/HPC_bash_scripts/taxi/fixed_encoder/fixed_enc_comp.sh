#!/bin/sh
#BSUB -J mlp_taxi
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -N
#BSUB -o ../logs/mlp_taxi/%J_var_test.out
#BSUB -e ../logs/mlp_taxi/%J_var_test.err

EXP_NAME=mlp_taxi/fixed_comp
mkdir "../models/${EXP_NAME}"


EPOCHS=200
KL_CYC=50
CUDA_DEVICE=0
BATCH_SIZE=200
BURN_IN_STEPS=48
SPLIT_LEN=60
EDGE_RATE=0.1
GUMBEL_TAU=0.5
LR=0.01
NLL_VARIANCE=0.005
SEED=44

PICKUP_DATA_PATH=taxi_data/full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=taxi_data/LGA_weather_full_2019.csv

LOCAL_ADJ_PATH=taxi_data/full_manhattan/full_year_full_manhattan_local_adj.npy
DTW_ADJ_PATH=taxi_data/full_manhattan/train_full_manhattan_dtw_adj_bin.npy
FULL_ADJ_PATH=taxi_data/full_manhattan/full_adj.npy
EMPTY_ADJ_PATH=taxi_data/full_manhattan/empty_adj.npy


ENC_N_HID=32
DEC_N_HID=32
DEC_MSG_HID=32
DEC_GRU_HID=32

# empty
python3 NRI_OD_train.py --experiment_name ${EXP_NAME}/empty_s${SEED} \
                        --epochs ${EPOCHS} \
                        --encoder_type fixed \
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
			--weight_decay 0.05 \
			--nll_variance ${NLL_VARIANCE} \
			--kl_cyc ${KL_CYC} \
			--lr ${LR} \
			--use_seed ${SEED} \
			--fixed_adj_matrix_path ${EMPTY_ADJ_PATH} \


# full		
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/full_s${SEED} \
                        --epochs ${EPOCHS} \
                        --encoder_type fixed \
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
			--weight_decay 0.05 \
			--nll_variance ${NLL_VARIANCE} \
			--kl_cyc ${KL_CYC} \
			--lr ${LR} \
			--use_seed ${SEED} \
			--fixed_adj_matrix_path ${FULL_ADJ_PATH} \

# local
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/local_s${SEED} \
                        --epochs ${EPOCHS} \
                        --encoder_type fixed \
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
			--weight_decay 0.05 \
			--nll_variance ${NLL_VARIANCE} \
			--kl_cyc ${KL_CYC} \
			--lr ${LR} \
			--use_seed ${SEED} \
			--fixed_adj_matrix_path ${LOCAL_ADJ_PATH} \
			

# dtw
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/dtw_s${SEED} \
                        --epochs ${EPOCHS} \
                        --encoder_type fixed \
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
			--weight_decay 0.05 \
			--nll_variance ${NLL_VARIANCE} \
			--kl_cyc ${KL_CYC} \
			--lr ${LR} \
			--use_seed ${SEED} \
			--fixed_adj_matrix_path ${DTW_ADJ_PATH} \
			

			

