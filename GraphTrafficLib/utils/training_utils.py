import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
import torch.nn.functional as F
from .losses import torch_nll_gaussian, kl_categorical
import numpy as np
import torch
from tqdm import tqdm


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
    n_nodes,
    gumbel_tau,
    gumbel_hard,
    use_weather
):
    nll_train = []
    kl_train = []
    mse_train = []
    mean_edge_prob_train = []

    encoder.train()
    decoder.train()

    for _, (data, weather) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        optimizer.zero_grad()

        data = data.cuda()

        if use_weather:
            weather = weather.cuda()
            logits = encoder(data, weather, rel_rec, rel_send)
        else:
            logits = encoder(data, rel_rec, rel_send)
        edges = F.gumbel_softmax(logits, tau=gumbel_tau, hard=gumbel_hard)  # RelaxedOneHotCategorical
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
            num_atoms=n_nodes,
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
        mean_edge_prob_train.append(edge_probs.mean(dim=(1, 0)).tolist())

    mse = np.mean(mse_train)
    nll = np.mean(nll_train)
    kl = np.mean(kl_train)
    mean_edge_prob = np.mean(np.array(mean_edge_prob_train), 0)
    return mse, nll, kl, mean_edge_prob


def val(
    encoder,
    decoder,
    val_dataloader,
    optimizer,
    rel_rec,
    rel_send,
    burn_in,
    burn_in_steps,
    split_len,
    log_prior,
    pred_steps,
    n_nodes,
    use_weather
):
    nll_val = []
    kl_val = []
    mse_val = []
    mean_edge_prob = []

    encoder.eval()
    decoder.eval()

    for _, (data, weather) in enumerate(tqdm(val_dataloader, desc="Validation", leave=False)):
        optimizer.zero_grad()
        with torch.no_grad():
            data = data.cuda()

            if use_weather:
                weather = weather.cuda()
                logits = encoder(data, weather, rel_rec, rel_send)
            else:
                logits = encoder(data, rel_rec, rel_send)

            edges = F.gumbel_softmax(logits, tau=0.01, hard=True)
            edge_probs = F.softmax(logits, dim=-1)
            mean_edge_prob.append(edge_probs.mean(dim=(1, 0)).tolist())

            pred_arr = decoder(
                data.transpose(1, 2),
                rel_rec,
                rel_send,
                edges,
                burn_in=burn_in,
                burn_in_steps=burn_in_steps,
                split_len=split_len,
            )
            pred = pred_arr.transpose(1, 2)[:, :, -pred_steps: ,:]
            target = data[:, :, -pred_steps:, :]

            loss_nll = torch_nll_gaussian(pred, target, 5e-5)
            loss_kl = kl_categorical(
                preds=edge_probs, log_prior=log_prior, num_atoms=n_nodes
            )  # Here I chose theirs since my implementation runs out of RAM :(
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())
        mse_val.append(F.mse_loss(pred, target).item())
    mse = np.mean(mse_val)
    nll = np.mean(nll_val)
    kl = np.mean(kl_val)
    mean_edge_prob = np.mean(np.array(mean_edge_prob), 0)
    return mse, nll, kl, mean_edge_prob


def dnri_train(
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
    n_nodes,
    gumbel_tau,
    gumbel_hard
):
    nll_train = []
    kl_train = []
    mse_train = []
    mean_edge_prob = []

    encoder.train()
    decoder.train()

    for _, (data, _) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        optimizer.zero_grad()

        data = data.cuda()

        _, posterior_logits, _ = encoder(data, rel_rec, rel_send)
        edges = F.gumbel_softmax(
            posterior_logits, tau=gumbel_tau, hard=gumbel_hard
        )  # RelaxedOneHotCategorical
        edge_probs = F.softmax(posterior_logits, dim=-1)
        mean_edge_prob.append(edge_probs.mean(dim=(1, 0)).tolist())

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
            num_atoms=n_nodes,
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
    mean_edge_prob = np.mean(np.array(mean_edge_prob), 0)
    return mse, nll, kl, mean_edge_prob


def dnri_val(
    encoder,
    decoder,
    val_dataloader,
    rel_rec,
    rel_send,
    burn_in,
    burn_in_steps,
    split_len,
    log_prior,
    n_nodes,
):
    nll_val = []
    kl_val = []
    mse_val = []
    mean_edge_prob = []

    encoder.eval()
    decoder.eval()

    for _, (data, _) in enumerate(tqdm(val_dataloader, desc="Validation", leave=False)):
        temp_mean_edge_probs = []
        with torch.no_grad():
            data = data.cuda()
            target = data[:, :, 1:, :]

            _, posterior_logits, prior_state = encoder(
                data[:, :, :burn_in_steps, :], rel_rec, rel_send
            )
            burn_in_edges = F.gumbel_softmax(
                posterior_logits, tau=0.01, hard=True
            )  # RelaxedOneHotCategorical
            burn_in_edge_probs = F.softmax(posterior_logits, dim=-1)

            data = data.transpose(1, 2)
            pred_all = []

            hidden = torch.autograd.Variable(
                torch.zeros(data.size(0), data.size(2), decoder.gru_hid)
            )
            edges = torch.autograd.Variable(
                torch.zeros(
                    burn_in_edges.size(0),
                    burn_in_edges.size(1),
                    data.size(1),
                    burn_in_edges.size(3),
                )
            )
            edge_probs = torch.autograd.Variable(
                torch.zeros(
                    burn_in_edges.size(0),
                    burn_in_edges.size(1),
                    data.size(1),
                    burn_in_edges.size(3),
                )
            )

            if data.is_cuda:
                hidden = hidden.cuda()
                edges = edges.cuda()
                edge_probs = edge_probs.cuda()

            edges[:, :, :burn_in_steps, :] = burn_in_edges
            edge_probs[:, :, :burn_in_steps, :] = burn_in_edge_probs

            for step in range(0, data.shape[1] - 1):
                if burn_in:
                    if step <= burn_in_steps - 1:
                        ins = data[
                            :, step, :, :
                        ]  # obs step different here to be time dim
                    else:
                        ins = pred_all[step - 1]
                        prior_logits, prior_state = encoder.single_step_forward(
                            ins, rel_rec, rel_send, prior_state
                        )
                        edges[:, :, step : step + 1, :] = F.gumbel_softmax(
                            prior_logits, tau=0.01, hard=True
                        )  # RelaxedOneHotCategorical
                        edge_probs[:, :, step : step + 1, :] = F.softmax(
                            prior_logits, dim=-1
                        )


                pred, hidden = decoder.do_single_step_forward(
                    ins, rel_rec, rel_send, edges, hidden, step
                )
                pred_all.append(pred)

            pred_arr = torch.stack(pred_all, dim=1)

            pred = pred_arr.transpose(1, 2).contiguous()

            loss_nll = torch_nll_gaussian(pred, target, 5e-5)
            loss_kl = kl_categorical(
                preds=edge_probs, log_prior=log_prior, num_atoms=n_nodes
            )  # Here I chose theirs since my implementation runs out of RAM :(
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())
        mse_val.append(F.mse_loss(pred, target).item())
        mse = np.mean(mse_val)
        nll = np.mean(nll_val)
        kl = np.mean(kl_val)
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


def val_lstm(
    model,
    val_dataloader,
    optimizer,
    burn_in,
    burn_in_steps,
    split_len,
):

    mse_val = []

    model.eval()

    for _, (data, _) in enumerate(val_dataloader):
        optimizer.zero_grad()
        with torch.no_grad():
            data = data.cuda()
            burn_in_data = data[:, :, :burn_in_steps, :].reshape(-1, burn_in_steps, 1)
            target = data[:, :, burn_in_steps:, :]
            pred = model(x=burn_in_data, pred_steps=split_len - burn_in_steps).reshape(
                target.shape
            )

            mse_val.append(F.mse_loss(pred, target).item())
        mse = np.mean(mse_val)
    return mse


def gumbel_tau_scheduler(tau_0, tau_end, epoch, n_epochs):
    slope = (tau_0 - tau_end) / n_epochs
    tau_cur = tau_0 - epoch * slope
    return tau_cur