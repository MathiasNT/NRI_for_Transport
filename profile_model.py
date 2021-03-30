from GraphTrafficLib.train import Trainer
import torch
import argparse


# Training settings
batch_size = 25
n_epochs = 200
dropout_p = 0
shuffle_train = True
shuffle_test = False

# Model settings
encoder_factor = True

# Data settings
normalize = True
train_frac = 0.8

# Model settings
burn_in_steps = 30
split_len = 40
pred_steps = split_len - burn_in_steps
encoder_steps = split_len

burn_in = True
kl_frac = 1
kl_free_bits_bound = 5
assert burn_in_steps + pred_steps == split_len

# Net sizes
# Encoder
enc_n_in = encoder_steps * 1
enc_n_hid = 128
enc_n_out = 2

# Decoder
dec_n_hid = 16
dec_n_out = 1
dec_f_in = 1
dec_msg_hid = 8
dec_msg_out = 8
dec_gru_hid = 8
dec_edge_types = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Parse args
    # General args
    parser.add_argument("--experiment_name", help="Name used for saving", required=True)

    # Cuda args
    parser.add_argument(
        "--cuda_device", type=int, default=1, help="Which cuda device to run on"
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs")

    args = parser.parse_args()

    print("Starting")

    print(f"Selecting GPU {args.cuda_device}")
    torch.cuda.set_device(args.cuda_device)
    torch.cuda.current_device()

    print(f"Running {args.epochs} epochs")
    trainer = Trainer(
        batch_size=batch_size,
        n_epochs=args.epochs,
        dropout_p=dropout_p,
        shuffle_train=shuffle_train,
        shuffle_test=shuffle_test,
        encoder_factor=encoder_factor,
        experiment_name=args.experiment_name,
        normalize=normalize,
        train_frac=train_frac,
        burn_in_steps=burn_in_steps,
        split_len=split_len,
        burn_in=burn_in,  # maybe remove this
        kl_frac=kl_frac,
        enc_n_hid=enc_n_hid,
        enc_n_out=enc_n_out,
        dec_n_hid=dec_n_hid,
        dec_n_out=dec_n_out,
        dec_f_in=dec_f_in,
        dec_msg_hid=dec_msg_hid,
        dec_msg_out=dec_msg_out,
        dec_gru_hid=dec_gru_hid,
        dec_edge_types=dec_edge_types,
    )
    print("Initialized")
    print("Profiling")
    trainer.profile_model()
