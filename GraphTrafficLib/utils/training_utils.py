import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
import torch.nn.functional as F
from .losses import torch_nll_gaussian, kl_categorical
import numpy as np
import torch
from tqdm import tqdm
from GraphTrafficLib.utils.data_utils.data_preprocess import ha_batch_renormalization, restandardize_data


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
    subset_dim = None
):
    nll = 0
    kl = 0
    mse = 0
    rmse = 0
    steps = 0
    mean_edge_prob_train = []

    encoder.train()
    decoder.train()

    for _, (data, weather, idxs) in enumerate(
        tqdm(train_dataloader, desc="Training", leave=False)
    ):
        optimizer.zero_grad()
        steps += len(data)
        data = data.cuda()
        #idxs = idxs.cuda()

        if use_weather:
            weather = weather.cuda()
            logits = encoder(data[:, :, :burn_in_steps, :], weather, rel_rec, rel_send)
        else:
            logits = encoder(data[:, :, :burn_in_steps, :], rel_rec, rel_send)
        edges = F.gumbel_softmax(
            logits, tau=gumbel_tau, hard=gumbel_hard
        )  # RelaxedOneHotCategorical
        edge_probs = F.softmax(logits, dim=-1)

        if subset_dim is not None:
            data = data[..., subset_dim ].unsqueeze(-1)
            norm_mean_tmp = norm_mean[..., subset_dim].unsqueeze(-1)
            norm_std_tmp = norm_std[..., subset_dim].unsqueeze(-1)

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
        pred = pred_arr.transpose(1, 2)[:, :, -pred_steps:, :]  # TODO .contiguous?
        target = data[:, :, -pred_steps:, :]
        
        loss_nll = torch_nll_gaussian(pred, target, variance=nll_variance)
        loss_kl = kl_categorical(
            preds=edge_probs,
            log_prior=log_prior,
            num_atoms=n_nodes,
        )
        # loss_mse = F.mse_loss(pred, target)

        if loss_type == "nll":
            loss = loss_nll + kl_frac * loss_kl
        # elif loss_type == "mse":
        #    loss = loss_mse + kl_frac * loss_kl

        loss.backward()
        optimizer.step()

        nll += loss_nll.detach() * len(data)
        kl += loss_kl.detach() * len(data)


        pred_idxs = idxs[:,-pred_steps:]
        if normalization == "ha":
            renormalized_pred = ha_batch_renormalization(batch=pred, batch_idxs=pred_idxs, datetime_list=time_list, mean_matrix=norm_mean_tmp, std_matrix=norm_std_tmp)
            renormalized_target = ha_batch_renormalization(batch=target, batch_idxs=pred_idxs, datetime_list=time_list, mean_matrix=norm_mean_tmp, std_matrix=norm_std_tmp)
        elif normalization == "z":
            renormalized_pred = restandardize_data(data=pred, data_mean=norm_mean_tmp, data_std=norm_std_tmp)
            renormalized_target = restandardize_data(data=target, data_mean=norm_mean_tmp, data_std=norm_std_tmp)

        
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
    subset_dim=None
):
    nll = 0
    kl = 0
    mse = 0
    steps = 0
    mean_edge_prob = []

    encoder.eval()
    decoder.eval()

    for _, (data, weather, idxs) in enumerate(
        tqdm(val_dataloader, desc="Validation", leave=False)
    ):
        optimizer.zero_grad()
        with torch.no_grad():
            data = data.cuda()
            steps += len(data)

            if use_weather:
                weather = weather.cuda()
                logits = encoder(
                    data[:, :, :burn_in_steps, :], weather, rel_rec, rel_send
                )
            else:
                logits = encoder(data[:, :, :burn_in_steps, :], rel_rec, rel_send)

            edges = F.gumbel_softmax(logits, tau=0.01, hard=True)
            edge_probs = F.softmax(logits, dim=-1)
            mean_edge_prob.append(edge_probs.mean(dim=(1, 0)).tolist())
                
            if subset_dim is not None:
                data = data[..., subset_dim ].unsqueeze(-1)
                norm_mean_tmp = norm_mean[..., subset_dim].unsqueeze(-1)
                norm_std_tmp = norm_std[..., subset_dim].unsqueeze(-1)

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
            loss_kl = kl_categorical(
                preds=edge_probs, log_prior=log_prior, num_atoms=n_nodes
            )  # Here I chose theirs since my implementation runs out of RAM :(
        nll += loss_nll.detach() * len(data)
        kl += loss_kl.detach() * len(data)


        pred_idxs = idxs[:,-pred_steps:]

        if normalization == "ha":
            renormalized_pred = ha_batch_renormalization(batch=pred, batch_idxs=pred_idxs, datetime_list=time_list, mean_matrix=norm_mean_tmp, std_matrix=norm_std_tmp)
            renormalized_target = ha_batch_renormalization(batch=target, batch_idxs=pred_idxs, datetime_list=time_list, mean_matrix=norm_mean_tmp, std_matrix=norm_std_tmp)
        elif normalization == "z":
            renormalized_pred = restandardize_data(data=pred, data_mean=norm_mean_tmp, data_std=norm_std_tmp)
            renormalized_target = restandardize_data(data=target, data_mean=norm_mean_tmp, data_std=norm_std_tmp)


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
    gumbel_hard,
):
    nll = 0
    kl = 0
    mse = 0
    rmse = 0
    steps = 0
    mean_edge_prob = []

    encoder.train()
    decoder.train()

    for _, (data, _) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        optimizer.zero_grad()
        steps += len(data)
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

        nll += loss_nll.detach() * len(data)
        kl += loss_kl.detach() * len(data)

        mse_batch = F.mse_loss(pred, target).detach()
        mse += mse_batch * len(data)
        rmse += mse_batch ** 0.5 * len(data)
    mse = mse / steps
    kl = kl / steps
    nll = nll / steps
    rmse = rmse / steps
    mean_edge_prob = np.mean(np.array(mean_edge_prob), 0)
    return mse, rmse, nll, kl, mean_edge_prob


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
    nll = 0
    kl = 0
    mse = 0
    rmse = 0
    steps = 0
    mean_edge_prob = []

    encoder.eval()
    decoder.eval()

    for _, (data, _) in enumerate(tqdm(val_dataloader, desc="Validation", leave=False)):
        temp_mean_edge_probs = []
        with torch.no_grad():
            steps += len(data)
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
        nll += loss_nll.detach() * len(data)
        kl += loss_kl.detach() * len(data)

        mse_batch = F.mse_loss(pred, target).detach()
        mse += mse_batch * len(data)
        rmse += mse_batch ** 0.5 * len(data)

    mse = mse / steps
    kl = kl / steps
    nll = nll / steps
    rmse = rmse / steps
    return mse, rmse, nll, kl


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

        mse_train.append(loss.detach())
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

            mse_val.append(F.mse_loss(pred, target).detach())
        mse = np.mean(mse_val)
    return mse


def gumbel_tau_scheduler(tau_0, tau_end, epoch, n_epochs):
    slope = (tau_0 - tau_end) / n_epochs
    tau_cur = tau_0 - epoch * slope
    return tau_cur