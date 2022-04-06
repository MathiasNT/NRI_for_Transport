#!/bin/sh
#BSUB -J learned_pems
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -N
#BSUB -o ../logs/learned_pems/%J_Output_learned_pems.out
#BSUB -e ../logs/learned_pems/%J_Error_learned_pems.err

EXP_NAME=learned_adj_pems
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
                        --encoder_type learned_adj \
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
			--lr ${LR} \
			--fixed_adj_matrix_path ${PRIOR_ADJ_PATH} \
