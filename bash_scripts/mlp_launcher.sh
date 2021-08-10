EXP_NAME=mlp_run_1
mkdir "../models/${EXP_NAME}"


EPOCHS=300
KL_CYC=100
CUDA_DEVICE=1
BATCH_SIZE=150 # above 100 is possible
BURN_IN_STEPS=30
SPLIT_LEN=40
PICKUP_DATA_PATH=full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv
      
EDGE_RATE=0.1

ENC_N_HID=128

DEC_N_HID=16
DEC_MSG_HID=8
DEC_GRU_HID=8



python NRI_OD_train.py  --experiment_name ${EXP_NAME}/mlp_2d_${EPOCHS}e50c \
                        --epochs $EPOCHS \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type mlp \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name ${PICKUP_DATA_PATH} \
                        --weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len  ${SPLIT_LEN}\
                        --edge_rate ${EDGE_RATE} \
                        --enc_n_hid ${ENC_N_HID} \
                        --dec_n_hid ${DEC_N_HID} \
                        --dec_msg_hid ${DEC_MSG_HID} \
                        --dec_gru_hid ${DEC_GRU_HID} \
