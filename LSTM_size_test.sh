EXP_NAME=LSTM_size_test
mkdir "../models/${EXP_NAME}"


EPOCHS=300
KL_CYC=50
CUDA_DEVICE=1
BATCH_SIZE=264
BURN_IN_STEPS=30
SPLIT_LEN=40
EDGE_RATE=0.1

PICKUP_DATA_PATH=split_manhattan/full_year_lower_manhattan_2d.npy
WEATHER_DATA_PATH=LGA_weather_full_2019.csv


LSTM_HID=256
python LSTM_train_script.py  \
   --experiment_name ${EXP_NAME}/LSTM_${LSTM_HID} \
   --cuda_device ${CUDA_DEVICE} \
   --epochs ${EPOCHS} \
   --pickup_data_name split_manhattan/full_year_lower_manhattan_2d.npy \
   --weather_data_name LGA_weather_full_2019.csv \
   --lstm_hid ${LSTM_HID} \
   --batch_size ${BATCH_SIZE} \

'''
LSTM_HID=128
python LSTM_train_script.py  \
   --experiment_name ${EXP_NAME}/LSTM_${LSTM_HID} \
   --cuda_device ${CUDA_DEVICE} \
   --epochs ${EPOCHS} \
   --pickup_data_name split_manhattan/full_year_lower_manhattan_2d.npy \
   --weather_data_name LGA_weather_full_2019.csv \
   --lstm_hid ${LSTM_HID} \
   --batch_size ${BATCH_SIZE} \

   
LSTM_HID=64
python LSTM_train_script.py  \
   --experiment_name ${EXP_NAME}/LSTM_${LSTM_HID} \
   --cuda_device ${CUDA_DEVICE} \
   --epochs ${EPOCHS} \
   --pickup_data_name split_manhattan/full_year_lower_manhattan_2d.npy \
   --weather_data_name LGA_weather_full_2019.csv \
   --lstm_hid ${LSTM_HID} \
   --batch_size ${BATCH_SIZE} \


LSTM_HID=32
python LSTM_train_script.py  \
   --experiment_name ${EXP_NAME}/LSTM_${LSTM_HID} \
   --cuda_device ${CUDA_DEVICE} \
   --epochs ${EPOCHS} \
   --pickup_data_name split_manhattan/full_year_lower_manhattan_2d.npy \
   --weather_data_name LGA_weather_full_2019.csv \
   --lstm_hid ${LSTM_HID} \
   --batch_size ${BATCH_SIZE} \


LSTM_HID=16
python LSTM_train_script.py  \
   --experiment_name ${EXP_NAME}/LSTM_${LSTM_HID} \
   --cuda_device ${CUDA_DEVICE} \
   --epochs ${EPOCHS} \
   --pickup_data_name split_manhattan/full_year_lower_manhattan_2d.npy \
   --weather_data_name LGA_weather_full_2019.csv \
   --lstm_hid ${LSTM_HID} \
   --batch_size ${BATCH_SIZE} \
'''