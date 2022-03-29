###
# A script for running a full NRI model on the NY bike data
###

EXP_NAME=final_models/bike
mkdir "../models/${EXP_NAME}"


EPOCHS=300
KL_CYC=50
CUDA_DEVICE=0
BATCH_SIZE=16
BURN_IN_STEPS=12
SPLIT_LEN=24
PRED_STEPS=12
EDGE_RATE=0.1
GUMBEL_TAU=0.5
N_EDGE_TYPES=2
LR=0.0005
WEIGHT_DECAY=0.0001
NLL_VARIANCE=0.0005


PICKUP_DATA_PATH=bike_data/
WEATHER_DATA_PATH=bike_data/bike_weather.csv
PRIOR_ADJ_PATH=bike_data/bike_adj_bin.npy



ENC_N_HID=64
DEC_N_HID=64
DEC_MSG_HID=64
DEC_GRU_HID=64

SEED=42
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/mlp_full_prior_lr${LR}_s${SEED}\
                        --epochs ${EPOCHS} \
                        --encoder_type mlp \
                        --loss_type nll \
			--cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
			--weather_data_name ${WEATHER_DATA_PATH} \
                        --batch_size ${BATCH_SIZE} \
                        --burn_in_steps ${BURN_IN_STEPS} \
                        --split_len ${SPLIT_LEN} \
			--pred_steps ${PRED_STEPS} \
                        --edge_rate ${EDGE_RATE} \
			--n_edge_types ${N_EDGE_TYPES} \
                        --enc_n_hid ${ENC_N_HID} \
			--dec_n_hid ${DEC_N_HID} \
			--dec_msg_hid ${DEC_MSG_HID} \
			--dec_gru_hid ${DEC_GRU_HID} \
			--gumbel_tau ${GUMBEL_TAU} \
			--gumbel_hard \
			--weight_decay ${WEIGHT_DECAY} \
			--nll_variance ${NLL_VARIANCE} \
			--prior_adj_path ${PRIOR_ADJ_PATH} \
			--use_seed ${SEED} \
			--use_weather \
			--lr ${LR}
