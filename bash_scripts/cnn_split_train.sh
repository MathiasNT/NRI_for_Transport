EXP_NAME=cnn_run_1
mkdir "../models/${EXP_NAME}"


EPOCHS=300
KL_CYC=50
CUDA_DEVICE=0
BATCH_SIZE=20
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.1
ENCODER_LR_RATE=0.1
N_EDGE_TYPES=2

PICKUP_DATA_PATH=full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv

ENC_N_HID=96
python NRI_OD_train.py  --experiment_name ${EXP_NAME}/cnn_hid${ENC_N_HID} \
                        --epochs ${EPOCHS} \
                        --kl_cyc ${KL_CYC} \
                        --encoder_type cnn \
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
                        --enc_n_hid ${ENC_N_HID}
