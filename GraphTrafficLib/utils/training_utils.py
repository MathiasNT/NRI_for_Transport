from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from .loss_utils import torch_nll_gaussian, kl_categorical
from .normalization_utils import (
    ha_batch_renormalization,
    restandardize_data,
)


def pretrain_encoder_epoch(
    encoder,
    train_dataloader,
    optimizer,
    n_nodes,
    log_prior,
    rel_rec,
    rel_send,
    use_weather,
    burn_in_steps,
):
    kl = 0
    steps = 0
    for _, (data, weather, _) in enumerate(
        tqdm(train_dataloader, desc="Pretrain encoder", leave=False)
    ):
        optimizer.zero_grad()
        steps += len(data)

        data = data.cuda()

        if use_weather:
            weather = weather.cuda()
            logits = encoder(data[:, :, :burn_in_steps, :], weather, rel_rec, rel_send)
        else:
            logits = encoder(data[:, :, :burn_in_steps, :], rel_rec, rel_send)

        edge_probs = F.softmax(logits, dim=-1)

        loss_kl = kl_categorical(
            preds=edge_probs,
            log_prior=log_prior,
            num_atoms=n_nodes,
        )

        loss_kl.backward()
        optimizer.step()

        kl += loss_kl.detach() * len(data)

    kl = kl / steps
    return kl


def train(
    encoder,
    decoder,
    train_dataloader,
    time_list,
    norm_mean,
    norm_std,
    normalization,
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
    use_weather,
    nll_variance,
    subset_dim=None,
):
    nll = 0
    kl = 0
    mse = 0
    rmse = 0
    steps = 0
    mean_edge_prob_train = []

    encoder.train()
    decoder.train()

    for _, (data, weather, idxs) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        optimizer.zero_grad()
        steps += len(data)
        data = data.cuda()

        if use_weather:
            weather = weather.cuda()
            logits = encoder(data[:, :, :burn_in_steps, :], weather, rel_rec, rel_send)
        else:
            logits = encoder(data[:, :, :burn_in_steps, :], rel_rec, rel_send)
        edges = F.gumbel_softmax(logits, tau=gumbel_tau, hard=gumbel_hard)
        edge_probs = F.softmax(logits, dim=-1)

        if subset_dim is not None:
            data = data[..., subset_dim].unsqueeze(-1)

        if use_weather:
            pred_arr = decoder(
                data.transpose(1, 2),
                weather,
                rel_rec,
                rel_send,
                edges,
                burn_in=burn_in,
                burn_in_steps=burn_in_steps,
                split_len=split_len,
            )
        else:
            pred_arr = decoder(
                data.transpose(1, 2),
                rel_rec,
                rel_send,
                edges,
                burn_in=burn_in,
                burn_in_steps=burn_in_steps,
                split_len=split_len,
            )
        pred = pred_arr.transpose(1, 2)[:, :, -pred_steps:, :]
        target = data[:, :, -pred_steps:, :]

        loss_nll = torch_nll_gaussian(pred, target, variance=nll_variance)
        loss_kl = kl_categorical(
            preds=edge_probs,
            log_prior=log_prior,
            num_atoms=n_nodes,
        )

        if loss_type == "nll":
            loss = loss_nll + kl_frac * loss_kl

        loss.backward()
        optimizer.step()

        nll += loss_nll.detach() * len(data)
        kl += loss_kl.detach() * len(data)

        pred_idxs = idxs[:, -pred_steps:]
        if normalization == "ha":
            renormalized_pred = ha_batch_renormalization(
                batch=pred,
                batch_idxs=pred_idxs,
                datetime_list=time_list,
                mean_matrix=norm_mean,
                std_matrix=norm_std,
            )
            renormalized_target = ha_batch_renormalization(
                batch=target,
                batch_idxs=pred_idxs,
                datetime_list=time_list,
                mean_matrix=norm_mean,
                std_matrix=norm_std,
            )
        elif normalization == "z":
            renormalized_pred = restandardize_data(
                data=pred, data_mean=norm_mean, data_std=norm_std
            )
            renormalized_target = restandardize_data(
                data=target, data_mean=norm_mean, data_std=norm_std
            )

        mse_batch = F.mse_loss(
            input=renormalized_pred[:, :, -(split_len - burn_in_steps) :, :],
            target=renormalized_target[:, :, -(split_len - burn_in_steps) :, :],
        ).detach()
        mse += mse_batch * len(data)

        mean_edge_prob_train.append(edge_probs.mean(dim=(1, 0)).tolist())

    mse = mse / steps
    kl = kl / steps
    nll = nll / steps
    rmse = mse ** 0.5

    mean_edge_prob = np.mean(np.array(mean_edge_prob_train), 0)
    return mse, rmse, nll, kl, mean_edge_prob


def val(
    encoder,
    decoder,
    val_dataloader,
    time_list,
    norm_mean,
    norm_std,
    normalization,
    optimizer,
    rel_rec,
    rel_send,
    burn_in,
    burn_in_steps,
    split_len,
    log_prior,
    pred_steps,
    n_nodes,
    use_weather,
    nll_variance,
    subset_dim=None,
):
    nll = 0
    kl = 0
    mse = 0
    steps = 0
    mean_edge_prob = []

    encoder.eval()
    decoder.eval()

    for _, (data, weather, idxs) in enumerate(tqdm(val_dataloader, desc="Validation", leave=False)):
        optimizer.zero_grad()
        with torch.no_grad():
            data = data.cuda()
            steps += len(data)

            if use_weather:
                weather = weather.cuda()
                logits = encoder(data[:, :, :burn_in_steps, :], weather, rel_rec, rel_send)
            else:
                logits = encoder(data[:, :, :burn_in_steps, :], rel_rec, rel_send)

            edges = F.gumbel_softmax(logits, tau=0.01, hard=True)
            edge_probs = F.softmax(logits, dim=-1)
            mean_edge_prob.append(edge_probs.mean(dim=(1, 0)).tolist())

            if subset_dim is not None:
                data = data[..., subset_dim].unsqueeze(-1)

            if use_weather:
                pred_arr = decoder(
                    data.transpose(1, 2),
                    weather,
                    rel_rec,
                    rel_send,
                    edges,
                    burn_in=burn_in,
                    burn_in_steps=burn_in_steps,
                    split_len=split_len,
                )
            else:
                pred_arr = decoder(
                    data.transpose(1, 2),
                    rel_rec,
                    rel_send,
                    edges,
                    burn_in=burn_in,
                    burn_in_steps=burn_in_steps,
                    split_len=split_len,
                )
            pred = pred_arr.transpose(1, 2)[:, :, -pred_steps:, :]
            target = data[:, :, -pred_steps:, :]

            loss_nll = torch_nll_gaussian(pred, target, nll_variance)
            loss_kl = kl_categorical(preds=edge_probs, log_prior=log_prior, num_atoms=n_nodes)
        nll += loss_nll.detach() * len(data)
        kl += loss_kl.detach() * len(data)

        pred_idxs = idxs[:, -pred_steps:]

        if normalization == "ha":
            renormalized_pred = ha_batch_renormalization(
                batch=pred,
                batch_idxs=pred_idxs,
                datetime_list=time_list,
                mean_matrix=norm_mean,
                std_matrix=norm_std,
            )
            renormalized_target = ha_batch_renormalization(
                batch=target,
                batch_idxs=pred_idxs,
                datetime_list=time_list,
                mean_matrix=norm_mean,
                std_matrix=norm_std,
            )
        elif normalization == "z":
            renormalized_pred = restandardize_data(
                data=pred, data_mean=norm_mean, data_std=norm_std
            )
            renormalized_target = restandardize_data(
                data=target, data_mean=norm_mean, data_std=norm_std
            )

        mse_batch = F.mse_loss(
            input=renormalized_pred[:, :, -(split_len - burn_in_steps) :, :],
            target=renormalized_target[:, :, -(split_len - burn_in_steps) :, :],
        ).detach()
        mse += mse_batch * len(data)
    mse = mse / steps
    kl = kl / steps
    nll = nll / steps
    rmse = mse ** 0.5
    mean_edge_prob = np.mean(np.array(mean_edge_prob), 0)
    return mse, rmse, nll, kl, mean_edge_prob


def gumbel_tau_scheduler(tau_0, tau_end, epoch, n_epochs):
    slope = (tau_0 - tau_end) / n_epochs
    tau_cur = tau_0 - epoch * slope
    return tau_cur


def cyc_anneal(epoch, cyclic):
    """
    returns anneal weight to multiply KLD with in elbo
    takes half a cycle to get to 1. for the rest of the cycle it will remain 1
    so it resembles https://github.com/haofuml/cyclical_annealing
    Function assumes epoch starts at 0
    """

    cycle = (epoch) % cyclic
    anneal = min(1, 2 / cyclic * cycle + 0.1)

    return anneal


def cyc_anneal_delayed(epoch, cyclic, delay):
    """
    returns anneal weight to multiply KLD with in elbo
    takes half a cycle to get to 1. for the rest of the cycle it will remain 1
    so it resembles https://github.com/haofuml/cyclical_annealing
    Function assumes epoch starts at 0
    """
    if epoch < delay:
        return 1

    cycle = (epoch) % cyclic
    anneal = min(1, 2 / cyclic * cycle + 0.1)

    return anneal
