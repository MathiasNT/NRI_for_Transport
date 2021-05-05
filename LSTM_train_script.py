from GraphTrafficLib.train import SimpleLSTMTrainer
import torch
import argparse


# Training settings
batch_size = 25
n_epochs = 200
dropout_p = 0
shuffle_train = True
shuffle_test = False

# Data settings
normalize = True
train_frac = 0.8

# Model settings
burn_in_steps = 30
split_len = 40
pred_steps = split_len - burn_in_steps
encoder_steps = split_len

burn_in = True
assert burn_in_steps + pred_steps == split_len

# Net sizes
# LSTM
lstm_hid = 128
lstm_dropout = 0

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
    print(f"Args are {args}")

    print("Starting")

    print(f"Selecting GPU {args.cuda_device}")
    torch.cuda.set_device(args.cuda_device)
    torch.cuda.current_device()

    print(f"Running {args.epochs} epochs")
    trainer = SimpleLSTMTrainer(
        batch_size=batch_size,
        n_epochs=args.epochs,
        dropout_p=dropout_p,
        shuffle_train=shuffle_train,
        shuffle_test=shuffle_test,
        experiment_name=args.experiment_name,
        normalize=normalize,
        train_frac=train_frac,
        burn_in_steps=burn_in_steps,
        split_len=split_len,
        burn_in=burn_in,  # maybe remove this
        lstm_hid=lstm_hid,
        lstm_dropout=lstm_dropout,
    )
    print("Initialized")
    trainer.train()
    print("Training")
    trainer.save_model()
