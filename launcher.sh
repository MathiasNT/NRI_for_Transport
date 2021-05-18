EPOCHS="300"

python LSTM_train_script.py  \
    --experiment_name LSTM_2d_${EPOCHS}e \
    --cuda_device 1 \
    --epochs $EPOCHS \
    --pickup_data_name split_manhattan/full_year_lower_manhattan_2d.npy \
    --weather_data_name LGA_weather_full_2019.csv \

python NRI_OD_train.py  --experiment_name cnn_2d_${EPOCHS}e50c \
                        --epochs $EPOCHS \
                        --kl_cyc 50 \
                        --encoder_type cnn \
                        --loss_type nll \
                        --cuda_device 1 \
                        --pickup_data_name split_manhattan/full_year_lower_manhattan_2d.npy \
                        --weather_data_name LGA_weather_full_2019.csv \
                        --batch_size 25 \
                        --burn_in_steps 30 \
                        --split_len 40 \
                        --edge_rate 0.10 \
                        
python NRI_OD_train.py  --experiment_name mlp_2d_${EPOCHS}e50c \
                        --epochs $EPOCHS \
                        --kl_cyc 50 \
                        --encoder_type mlp \
                        --loss_type nll \
                        --cuda_device 1 \
                        --pickup_data_name split_manhattan/full_year_lower_manhattan_2d.npy \
                        --weather_data_name LGA_weather_full_2019.csv \
                        --batch_size 25 \
                        --burn_in_steps 30 \
                        --split_len 40 \
                        --edge_rate 0.10 \
