from GraphTrafficLib.train import Trainer

# Training settings
batch_size = 25
n_epochs = 100
dropout_p = 0
shuffle_train = True
shuffle_test = False

# Model settings
encoder_factor = True

# Saving settings
gru_dev_file_name = "test"

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
    trainer = Trainer(
        batch_size=batch_size,
        n_epochs=n_epochs,
        dropout_p=dropout_p,
        shuffle_train=shuffle_train,
        shuffle_test=shuffle_test,
        encoder_factor=encoder_factor,
        gru_dev_file_name=gru_dev_file_name,
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
    trainer.train()
    trainer.save_model()

