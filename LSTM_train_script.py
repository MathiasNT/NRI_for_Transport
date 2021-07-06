from GraphTrafficLib.train import SimpleLSTMTrainer
import torch
import argparse


# Training settings
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
    # Data args
    parser.add_argument(
        "--pickup_data_name", help="path from datafolder to pickupdata", required=True
    )
    parser.add_argument(
        "--dropoff_data_name", help="path from datafolder to dropoffdata"
    )
    parser.add_argument(
        "--weather_data_name", help="path from datafolder to weaher data", required=True
    )

    # General args
    parser.add_argument("--experiment_name", help="Name used for saving", required=True)

    # Cuda args
    parser.add_argument(
        "--cuda_device", type=int, default=1, help="Which cuda device to run on"
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs")
    parser.add_argument("--batch_size", type=int, help="The batch size, default 25")

    # Model args
    parser.add_argument("--lstm_hid", type=int, help="The hidden size of the LSTM model")

    args = parser.parse_args()

    dataset_folder = "../datafolder"
    proc_folder = f"{dataset_folder}/procdata"

    pickup_data_path = f"{proc_folder}/{args.pickup_data_name}"
    if args.dropoff_data_name is not None:
        dropoff_data_path = f"{proc_folder}/{args.dropoff_data_name}"
    else:
        dropoff_data_path = args.dropoff_data_name
    weather_data_path = f"{proc_folder}/{args.weather_data_name}"

    print(f"Args are {args}")

    print("Starting")

    print(f"Selecting GPU {args.cuda_device}")
    torch.cuda.set_device(args.cuda_device)
    torch.cuda.current_device()

    print(f"Running {args.epochs} epochs")
    trainer = SimpleLSTMTrainer(
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        dropout_p=dropout_p,
        shuffle_train=shuffle_train,
        shuffle_val=shuffle_test,
        experiment_name=args.experiment_name,
        normalize=normalize,
        train_frac=train_frac,
        burn_in_steps=burn_in_steps,
        split_len=split_len,
        burn_in=burn_in,  # maybe remove this
        lstm_hid=args.lstm_hid,
        lstm_dropout=lstm_dropout,
    )
    print("Initialized")
    trainer.load_data(
        data_path=pickup_data_path,
        dropoff_data_path=dropoff_data_path,
        weather_data_path=weather_data_path,
    )
    print("Data loaded")
    trainer.train()
    print("Training")
    trainer.save_model()
