EXP_NAME=long_test2
mkdir "../models/${EXP_NAME}"


EPOCHS=600
KL_CYC=50
CUDA_DEVICE=0
BATCH_SIZE=400
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.01

PICKUP_DATA_PATH=split_manhattan/full_year_lower_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv

python NRI_OD_train.py  --experiment_name ${EXP_NAME}/cnn_2d_${EPOCHS}e50c \
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
