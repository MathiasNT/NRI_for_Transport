EXP_NAME=rnn_debug_test
mkdir "../models/${EXP_NAME}"


EPOCHS=10
KL_CYC=50
CUDA_DEVICE=1
BATCH_SIZE=10
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.1
ENCODER_LR_RATE=0.1
ENC_N_HID=32
N_EDGE_TYPES=2

PICKUP_DATA_PATH=full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv


python NRI_OD_train.py  --experiment_name ${EXP_NAME}/gru_2d_${EPOCHS}e50c \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type gru \
                        --loss_type nll \
                        --cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
                        --weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
                        --edge_rate ${EDGE_RATE} \
                        --encoder_lr_frac ${ENCODER_LR_RATE} \
                        --n_edge_types ${N_EDGE_TYPES} \
                        --enc_n_hid ${ENC_N_HID} \