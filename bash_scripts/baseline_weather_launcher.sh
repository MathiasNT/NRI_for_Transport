EXP_NAME=baseline_weather_batch_size_test
mkdir "../models/${EXP_NAME}"

EPOCHS=5
KL_CYC=50
CUDA_DEVICE=1
BATCH_SIZE=350
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.1
#WEIGHT_DECAY=0.005


ENC_N_HID=128
DEC_N_HID=16
DEC_MSG_HID=8
DEC_GRU_HID=8

PICKUP_DATA_PATH=full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv
DTW_ADJ_PATH=full_manhattan/full_year_full_manhattan_dtw_adj_bin.npy
LOCAL_ADJ_PATH=full_manhattan/full_year_full_manhattan_local_adj.npy


#BATCH_SIZE=300
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/local_bs${BATCH_SIZE} \
                        --epochs ${EPOCHS} \
                        --encoder_type fixed \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
			--fixed_adj_matrix_path ${LOCAL_ADJ_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --enc_n_hid ${ENC_N_HID} \
			--dec_n_hid ${DEC_N_HID} \
			--dec_msg_hid ${DEC_MSG_HID} \
			--dec_gru_hid ${DEC_GRU_HID} \
			--use_weather \
			
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/local_bs${BATCH_SIZE} \
                        --epochs ${EPOCHS} \
                        --encoder_type fixed \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
			--fixed_adj_matrix_path ${DTW_ADJ_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --enc_n_hid ${ENC_N_HID} \
			--dec_n_hid ${DEC_N_HID} \
			--dec_msg_hid ${DEC_MSG_HID} \
			--dec_gru_hid ${DEC_GRU_HID} \
			--use_weather \