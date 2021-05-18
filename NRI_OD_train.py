from GraphTrafficLib.train import Trainer
import torch
import argparse


# Training settings
dropout_p = 0
shuffle_train = True
shuffle_test = False

# Model settings
encoder_factor = True

# Data settings
normalize = True
train_frac = 0.8

# Model settings
burn_in = True
kl_frac = 1

# Net sizes
# Encoder
enc_n_hid = 128
enc_n_out = 2

# Decoder
dec_n_hid = 16
dec_msg_hid = 8
dec_msg_out = 8
dec_gru_hid = 8
dec_edge_types = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Parse args
    # Data args
    # TODO fix the pickup data naming
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
    parser.add_argument(
        "--kl_cyc", type=int, help="The period for the cyclical annealing"
    )
    parser.add_argument("--batch_size", type=int, help="The batch size, default 25")

    # Model args
    parser.add_argument(
        "--encoder_type", help="which encoder type to use (cnn or mlp)", required=True
    )
    parser.add_argument(
        "--loss_type",
        help="Which loss to use 'nll' or 'mse' (both use KL aswell)",
        required=True,
    )
    parser.add_argument(
        "--edge_rate",
        help="The prior on the edge probabilities",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--burn_in_steps",
        type=int,
        help="The amount of burn in steps for the decoder",
        required=True,
    )
    parser.add_argument(
        "--split_len",
        type=int,
        help="The overall split len (burn_in_steps + pred_steps = split_len)",
        required=True,
    )

    args = parser.parse_args()

    pred_steps = args.split_len - args.burn_in_steps
    encoder_steps = args.split_len

    dataset_folder = "../datafolder"
    proc_folder = f"{dataset_folder}/procdata"

    pickup_data_path = f"{proc_folder}/{args.pickup_data_name}"
    if args.dropoff_data_name is not None:
        dropoff_data_path = f"{proc_folder}/{args.dropoff_data_name}"
        node_f_dim = 1
    else:
        dropoff_data_path = args.dropoff_data_name
        node_f_dim = 2
    weather_data_path = f"{proc_folder}/{args.weather_data_name}"

    print(f"Args are {args}")

    print("Starting")

    print(f"Selecting GPU {args.cuda_device}")
    torch.cuda.set_device(args.cuda_device)
    torch.cuda.current_device()

    print(f"Running {args.epochs} epochs")
    trainer = Trainer(
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        dropout_p=dropout_p,
        shuffle_train=shuffle_train,
        shuffle_test=shuffle_test,
        encoder_factor=encoder_factor,
        experiment_name=args.experiment_name,
        normalize=normalize,
        train_frac=train_frac,
        burn_in_steps=args.burn_in_steps,
        split_len=args.split_len,
        burn_in=burn_in,  # maybe remove this
        kl_frac=kl_frac,
        kl_cyc=args.kl_cyc,
        loss_type=args.loss_type,
        edge_rate=args.edge_rate,
        encoder_type=args.encoder_type,
        node_f_dim=node_f_dim,
        enc_n_hid=enc_n_hid,
        enc_n_out=enc_n_out,
        dec_n_hid=dec_n_hid,
        dec_msg_hid=dec_msg_hid,
        dec_msg_out=dec_msg_out,
        dec_gru_hid=dec_gru_hid,
        dec_edge_types=dec_edge_types,
    )
    print("Initialized")

    print(f"Loading data at {pickup_data_path}")
    trainer.load_data(
        data_path=pickup_data_path,
        dropoff_data_path=dropoff_data_path,
        weather_data_path=weather_data_path,
    )
    print("Data loaded")
    trainer.train()
    print("Training")
    trainer.save_model()
