##
# Script for running full NRI model with a local prior for the PEMS dataset
##

EXP_NAME=mlp_pems
mkdir "../models/${EXP_NAME}"


EPOCHS=30
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


PICKUP_DATA_PATH=pems_data/
PRIOR_ADJ_PATH=pems_data/approx_local_adj.npy

ENC_N_HID=32
DEC_N_HID=32
DEC_MSG_HID=32
DEC_GRU_HID=32
NODE_F_DIM=1

SEED=42
python3 NRI_OD_train.py  --experiment_name ${EXP_NAME}/mlp_lr${LR}_s${SEED}\
                        --epochs ${EPOCHS} \
                        --encoder_type mlp \
                        --loss_type nll \
			--cuda_device ${CUDA_DEVICE} \
                        --pickup_data_name  ${PICKUP_DATA_PATH}\
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
			--kl_cyc ${KL_CYC} \
			--use_seed ${SEED} \
			--node_f_dim ${NODE_F_DIM} \
			--lr ${LR}
