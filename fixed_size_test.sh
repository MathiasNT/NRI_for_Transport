EXP_NAME=batch_size_test
mkdir "../models/${EXP_NAME}"

EPOCHS=100
KL_CYC=50
CUDA_DEVICE=1
#BATCH_SIZE=400
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.1

PICKUP_DATA_PATH=split_manhattan/full_year_lower_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv

DEC_N_HID=8
DEC_MSG_HID=8
DEC_GRU_HID=16

BATCH_SIZE=600
python NRI_OD_train.py  --experiment_name ${EXP_NAME}/fixed_n_hid_${DEC_N_HID}_msg_${DEC_MSG_HID}_gru_${DEC_GRU_HID}_batch_${BATCH_SIZE} \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type fixed \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --fixed_adj_matrix_path split_manhattan/lower_local_adj.npy \
                        --dec_n_hid ${DEC_N_HID} \
                        --dec_msg_hid ${DEC_MSG_HID} \
                        --dec_gru_hid ${DEC_GRU_HID}


BATCH_SIZE=400
python NRI_OD_train.py  --experiment_name ${EXP_NAME}/fixed_n_hid_${DEC_N_HID}_msg_${DEC_MSG_HID}_gru_${DEC_GRU_HID}_batch_${BATCH_SIZE} \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type fixed \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --fixed_adj_matrix_path split_manhattan/lower_local_adj.npy \
                        --dec_n_hid ${DEC_N_HID} \
                        --dec_msg_hid ${DEC_MSG_HID} \
                        --dec_gru_hid ${DEC_GRU_HID}

BATCH_SIZE=200
python NRI_OD_train.py  --experiment_name ${EXP_NAME}/fixed_n_hid_${DEC_N_HID}_msg_${DEC_MSG_HID}_gru_${DEC_GRU_HID}_batch_${BATCH_SIZE} \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type fixed \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --fixed_adj_matrix_path split_manhattan/lower_local_adj.npy \
                        --dec_n_hid ${DEC_N_HID} \
                        --dec_msg_hid ${DEC_MSG_HID} \
                        --dec_gru_hid ${DEC_GRU_HID}

BATCH_SIZE=100
python NRI_OD_train.py  --experiment_name ${EXP_NAME}/fixed_n_hid_${DEC_N_HID}_msg_${DEC_MSG_HID}_gru_${DEC_GRU_HID}_batch_${BATCH_SIZE} \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type fixed \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --fixed_adj_matrix_path split_manhattan/lower_local_adj.npy \
                        --dec_n_hid ${DEC_N_HID} \
                        --dec_msg_hid ${DEC_MSG_HID} \
                        --dec_gru_hid ${DEC_GRU_HID}

BATCH_SIZE=50
python NRI_OD_train.py  --experiment_name ${EXP_NAME}/fixed_n_hid_${DEC_N_HID}_msg_${DEC_MSG_HID}_gru_${DEC_GRU_HID}_batch_${BATCH_SIZE} \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type fixed \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --fixed_adj_matrix_path split_manhattan/lower_local_adj.npy \
                        --dec_n_hid ${DEC_N_HID} \
                        --dec_msg_hid ${DEC_MSG_HID} \
                        --dec_gru_hid ${DEC_GRU_HID}

# python NRI_OD_train.py  --experiment_name ${EXP_NAME}/fixed_n_hid_${DEC_N_HID}_msg_${DEC_MSG_HID}_gru_${DEC_GRU_HID} \
#                         --epochs ${EPOCHS} \
#                         --kl_cyc ${KL_CYC} \
#                         --encoder_type fixed \
#                         --loss_type nll \
#                         --cuda_device ${CUDA_DEVICE} \
#                         --pickup_data_name  ${PICKUP_DATA_PATH}\
#                         --weather_data_name ${WEATHER_DATA_PATH} \
#                         --batch_size ${BATCH_SIZE} \
#                         --burn_in_steps ${BURN_IN_STEPS} \
#                         --split_len ${SPLIT_LEN} \
#                         --edge_rate ${EDGE_RATE} \
#                         --fixed_adj_matrix_path split_manhattan/lower_local_adj.npy \
#                         --dec_n_hid ${DEC_N_HID} \
#                         --dec_msg_hid ${DEC_MSG_HID} \
#                         --dec_gru_hid ${DEC_GRU_HID}


# python NRI_OD_train.py  --experiment_name ${EXP_NAME}/fixed_n_hid_${DEC_N_HID}_msg_${DEC_MSG_HID}_gru_${DEC_GRU_HID} \
#                         --epochs ${EPOCHS} \
#                         --kl_cyc ${KL_CYC} \
#                         --encoder_type fixed \
#                         --loss_type nll \
#                         --cuda_device ${CUDA_DEVICE} \
#                         --pickup_data_name  ${PICKUP_DATA_PATH}\
#                         --weather_data_name ${WEATHER_DATA_PATH} \
#                         --batch_size ${BATCH_SIZE} \
#                         --burn_in_steps ${BURN_IN_STEPS} \
#                         --split_len ${SPLIT_LEN} \
#                         --edge_rate ${EDGE_RATE} \
#                         --fixed_adj_matrix_path split_manhattan/lower_local_adj.npy \
#                         --dec_n_hid ${DEC_N_HID} \
#                         --dec_msg_hid ${DEC_MSG_HID} \
#                         --dec_gru_hid ${DEC_GRU_HID}
