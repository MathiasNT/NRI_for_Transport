from GraphTrafficLib.train import Trainer
import torch
import argparse


# Training settings
shuffle_train = True
shuffle_val = False

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
# enc_n_hid = 128


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
        "--weather_data_name", help="path from datafolder to weaher data"
    )

    # General args
    parser.add_argument("--experiment_name", help="Name used for saving", required=True)
    parser.add_argument("--checkpoint_path", help="Path to model experiment to load checkpoint from")
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
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
    parser.add_argument(
        "--encoder_lr_frac",
        type=float,
        help="The fraction with which the encoder lr should be smaller than decoder lr",
        default=1,
    )
    parser.add_argument("--lr_decay_step", help="How often to do lr decay", default=100)
    parser.add_argument("--lr_decay_gamma", help="Factor to decay lr with", default=0.5)
    parser.add_argument(
        "--no_bn",
        dest="use_bn",
        help="Whether or not to use bn in MLP modules",
        action="store_false",
    )
    parser.add_argument(
        "--gumbel_hard",
        action="store_true",
        default=False,
        help="Uses discrete sampling in training forward pass",
    )
    parser.add_argument(
        "--gumbel_tau",
        type=float,
        help="The tau value in the gumbel distribution",
        default=0.5,
    )
    parser.add_argument(
        "--gumbel_anneal",
        action="store_true",
        default=False,
        help="Whether to anneal the tau value in the gumbel distribution",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="The L2 regularization for the optimizer (default=0)",
        default=0,
    )
    parser.add_argument(
        "--dropout_p", type=float, default=0, help="Dropout rate (1-keep)"
    )
    parser.add_argument(
        "--nll_variance", type=float, default=5e-5, help="Variance for NLL loss"
    )

    # Model args
    parser.add_argument(
        "--encoder_type", help="which encoder type to use (cnn or mlp)", required=True
    )
    parser.add_argument(
        "--n_edge_types",
        help="The number of different edge types to model",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--enc_n_hid", help="The hidden dim of the encoder", type=int, default=128
    )
    parser.add_argument(
        "--init_weights",
        dest="init_weights",
        help="Whether to use special init for CNN weights",
        action="store_true",
    )
    parser.add_argument(
        "--dec_n_hid", help="Hidden size of out part of decoder", type=int, default=16
    )
    parser.add_argument(
        "--dec_msg_hid", help="Hidden size of message in decoder", type=int, default=8
    )
    parser.add_argument(
        "--dec_gru_hid",
        help="Hidden size of the recurrent state of the decoder",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--fixed_adj_matrix_path",
        help="Path to fixed adjacancy matrix for fixed encoder",
        required=False,
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
        "--prior_adj_path", help="path to adj matrix of prior", default=None, type=str
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
    parser.add_argument(
        "--use_weather",
        action="store_true",
        default=False,
        help="Whether to include weather in the encoder",
    )
    parser.add_argument("--node_f_dim", type=int, default=2, help="The amount of features on pr. timestep on nodes")
    parser.add_argument("--use_seed", type=int, help="Seed for torch RNG")

    args = parser.parse_args()

    if args.use_seed is not None:
        torch.manual_seed(args.use_seed)

    pred_steps = args.split_len - args.burn_in_steps
    encoder_steps = args.split_len

    node_f_dim = args.node_f_dim

    dataset_folder = "../datafolder"
    proc_folder = f"{dataset_folder}/procdata"

    if args.fixed_adj_matrix_path is not None:
        args.fixed_adj_matrix_path = f"{proc_folder}/{args.fixed_adj_matrix_path}"

    print(f"Args are {args}")

    print("Starting")

    print(f"Selecting GPU {args.cuda_device}")
    torch.cuda.set_device(args.cuda_device)
    torch.cuda.current_device()

    print(f"Running {args.epochs} epochs")
    trainer = Trainer(
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        dropout_p=args.dropout_p,
        shuffle_train=shuffle_train,
        shuffle_val=shuffle_val,
        lr=args.lr,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
        encoder_factor=encoder_factor,
        skip_first=True,
        experiment_name=args.experiment_name,
        normalize=normalize,
        train_frac=train_frac,
        burn_in_steps=args.burn_in_steps,
        split_len=args.split_len,
        burn_in=burn_in,
        kl_frac=kl_frac,
        kl_cyc=args.kl_cyc,
        loss_type=args.loss_type,
        edge_rate=args.edge_rate,
        encoder_type=args.encoder_type,
        node_f_dim=node_f_dim,
        enc_n_hid=args.enc_n_hid,
        rnn_enc_n_hid=None,
        n_edge_types=args.n_edge_types,
        dec_n_hid=args.dec_n_hid,
        dec_msg_hid=args.dec_msg_hid,
        dec_gru_hid=args.dec_gru_hid,
        fixed_adj_matrix_path=args.fixed_adj_matrix_path,
        encoder_lr_frac=args.encoder_lr_frac,
        use_bn=args.use_bn,
        init_weights=args.init_weights,
        gumbel_tau=args.gumbel_tau,
        gumbel_hard=args.gumbel_hard,
        gumbel_anneal=args.gumbel_anneal,
        weight_decay=args.weight_decay,
        use_weather=args.use_weather,
        nll_variance=args.nll_variance,
        prior_adj_path=args.prior_adj_path,
        checkpoint_path=args.checkpoint_path
    )

    print("Initialized")

    if args.pickup_data_name.split("_")[0] == "taxi":
        trainer.load_data(
            proc_folder=proc_folder,
            data_name=args.pickup_data_name,
            weather_data_name=args.weather_data_name,
        )
    elif args.pickup_data_name.split("_")[0] == "bike":
        trainer.load_data_bike(
            proc_folder=proc_folder,
            bike_folder=args.pickup_data_name,
            weather_data_path=args.weather_data_name,
        )
    elif args.pickup_data_name.split("_")[0] == "pems":
        trainer.load_data_road(
            proc_folder=proc_folder,
            road_folder=args.pickup_data_name,
        )
    else:
        raise NameError("data path is neither bike or taxi")
    print("Data loaded")
    trainer.train()
    print("Training")
    trainer.save_model()


# batch_size=25,
# n_epochs=100,
# dropout_p=0,
# shuffle_train=True,
# shuffle_val=False,
# encoder_factor=True,
# experiment_name="test",
# normalize=True,
# train_frac=0.8,
# burn_in_steps=30,
# split_len=40,
# burn_in=True,  # maybe remove this
# kl_frac=1,
# kl_cyc=None,
# loss_type=None,
# edge_rate=0.01,
# encoder_type="mlp",
# node_f_dim=1,
# enc_n_hid=128,
# rnn_enc_n_hid=None,
# n_edge_types=2,
# dec_n_hid=16,
# dec_msg_hid=8,
# dec_gru_hid=8,
# skip_first=True,
# lr=0.001,
# lr_decay_step=100,
# lr_decay_gamma=0.5,
# fixed_adj_matrix_path=None,
# encoder_lr_frac=1,
# use_bn=True,
# init_weights=False,
# gumbel_tau=0.5,
# gumbel_hard=True,
# gumbel_anneal=None,
# weight_decay=0,
# use_weather=False,
# nll_variance=5e-5