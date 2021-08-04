EXP_NAME=baseline_setup_test
mkdir "../models/${EXP_NAME}"


EPOCHS=1
KL_CYC=50
CUDA_DEVICE=1
BATCH_SIZE=100
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.1

PICKUP_DATA_PATH=full_manhattan/full_year_full_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv

LSTM_HID=16
python LSTM_train_script.py  \
    --experiment_name ${EXP_NAME}/LSTM_2d_${EPOCHS}e \
    --cuda_device ${CUDA_DEVICE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --pickup_data_name ${PICKUP_DATA_PATH} \
    --weather_data_name ${WEATHER_DATA_PATH} \
    --lstm_hid ${LSTM_HID} \

python NRI_OD_train.py  --experiment_name ${EXP_NAME}/fixed_local_2d_${EPOCHS}e50c \
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
                        --fixed_adj_matrix_path full_manhattan/full_year_full_manhattan_local_adj.npy                        

python NRI_OD_train.py  --experiment_name ${EXP_NAME}/fixed_dtw_2d_${EPOCHS}e50c \
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
                        --fixed_adj_matrix_path full_manhattan/full_year_full_manhattan_dtw_adj_bin.npy                        