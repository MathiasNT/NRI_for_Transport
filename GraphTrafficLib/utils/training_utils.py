import matplotlib.pyplot as plt
import torch.nn.functional as F
from .losses import torch_nll_gaussian, kl_categorical
import numpy as np
import torch


def plot_training(train_mse_arr, train_nll_arr, train_kl_arr, train_acc_arr):
    if len(train_mse_arr) <= 5:
        return
    fig, axs = plt.subplots(3, figsize=(25, 13))
    axs[0].plot(train_mse_arr[5:])
    axs[0].title.set_text("MSE")

    axs[1].plot(train_nll_arr[5:])
    axs[1].title.set_text("NLL")

    axs[2].plot(train_kl_arr[5:])
    axs[2].title.set_text("KL")

    # axs[3].plot(train_acc_arr[5:])
    # axs[3].title.set_text("Edge Acc")
    plt.show()


def train(
    encoder,
    decoder,
    train_dataloader,
    optimizer,
    rel_rec,
    rel_send,
    burn_in,
    burn_in_steps,
    split_len,
    log_prior,
    kl_frac,
    loss_type,
    pred_steps,
    skip_first,
):
    nll_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()

    for _, (data, _) in enumerate(train_dataloader):
        optimizer.zero_grad()

        data = data.cuda()

        logits = encoder(data, rel_rec, rel_send)
        edges = F.gumbel_softmax(logits, tau=0.5, hard=True)  # RelaxedOneHotCategorical
        edge_probs = F.softmax(logits, dim=-1)

        pred_arr = decoder(
            data.transpose(1, 2),
            rel_rec,
            rel_send,
            edges,
            burn_in=burn_in,
            burn_in_steps=burn_in_steps,
            split_len=split_len,
        )
        pred = pred_arr.transpose(1, 2)[:, :, -pred_steps:, :]  # TODO .contiguous?
        target = data[:, :, -pred_steps:, :]

        loss_nll = torch_nll_gaussian(pred, target, 5e-5)
        loss_kl = kl_categorical(
            preds=edge_probs,
            log_prior=log_prior,
            num_atoms=132,  # Watch out for hardcode!! TODO
        )  # Here I chose theirs since my implementation runs out of RAM :(
        loss_mse = F.mse_loss(pred, target)

        if loss_type == "nll":
            loss = loss_nll + kl_frac * loss_kl
        elif loss_type == "mse":
            loss = loss_mse + kl_frac * loss_kl

        loss.backward()
        optimizer.step()

        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        mse_train.append(F.mse_loss(pred, target).item())
    mse = np.mean(mse_train)
    nll = np.mean(nll_train)
    kl = np.mean(kl_train)
    return mse, nll, kl


def test(
    encoder,
    decoder,
    test_dataloader,
    optimizer,
    rel_rec,
    rel_send,
    burn_in,
    burn_in_steps,
    split_len,
    log_prior,
):
    nll_test = []
    kl_test = []
    mse_test = []

    encoder.eval()
    decoder.eval()

    for batch_idx, (data, weather) in enumerate(test_dataloader):
        optimizer.zero_grad()
        with torch.no_grad():
            data = data.cuda()

            logits = encoder(data, rel_rec, rel_send)
            edges = F.gumbel_softmax(logits, tau=0.5, hard=True)
            edge_probs = F.softmax(logits, dim=-1)

            pred_arr = decoder(
                data.transpose(1, 2),
                rel_rec,
                rel_send,
                edges,
                burn_in=burn_in,
                burn_in_steps=burn_in_steps,
                split_len=split_len,
            )
            pred = pred_arr.transpose(1, 2).contiguous()
            target = data[:, :, 1:, :]

            loss_nll = torch_nll_gaussian(pred, target, 5e-5)
            loss_kl = kl_categorical(
                preds=edge_probs, log_prior=log_prior, num_atoms=132
            )  # Here I chose theirs since my implementation runs out of RAM :(
        nll_test.append(loss_nll.item())
        kl_test.append(loss_kl.item())
        mse_test.append(F.mse_loss(pred, target).item())
        mse = np.mean(mse_test)
        nll = np.mean(nll_test)
        kl = np.mean(kl_test)
    return mse, nll, kl


def train_lstm(model, train_dataloader, optimizer, burn_in, burn_in_steps, split_len):

    mse_train = []

    model.train()
    for _, (data, _) in enumerate(train_dataloader):
        optimizer.zero_grad()

        data = data.cuda()
        burn_in_data = data[:, :, :burn_in_steps, :].reshape(-1, burn_in_steps, 1)
        target = data[:, :, burn_in_steps:, :]
        pred = model(x=burn_in_data, pred_steps=split_len - burn_in_steps).reshape(
            target.shape
        )

        loss = F.mse_loss(pred, target)

        loss.backward()
        optimizer.step()

        mse_train.append(loss.item())
    mse = np.mean(mse_train)
    return mse


def test_lstm(
    model,
    test_dataloader,
    optimizer,
    burn_in,
    burn_in_steps,
    split_len,
):

    mse_test = []

    model.eval()

    for _, (data, weather) in enumerate(test_dataloader):
        optimizer.zero_grad()
        with torch.no_grad():
            data = data.cuda()
            burn_in_data = data[:, :, :burn_in_steps, :].reshape(-1, burn_in_steps, 1)
            target = data[:, :, burn_in_steps:, :]
            pred = model(x=burn_in_data, pred_steps=split_len - burn_in_steps).reshape(
                target.shape
            )

            mse_test.append(F.mse_loss(pred, target).item())
        mse = np.mean(mse_test)
    return mse