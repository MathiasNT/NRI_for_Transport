from tqdm import tqdm
import torch

def create_lag1_and_ha_predictions(
    train_dataloader,
    test_dataloader,
    burn_in,
    burn_in_steps,
    split_len,
):
    y_true = []
    y_lag1 = []
    for i, (data, weather) in tqdm(enumerate(test_dataloader)):

        y_true.append(data[:, :, burn_in_steps, :].squeeze())
        y_lag1.append(data[:, :, burn_in_steps - 1, :].squeeze())

    y_true = torch.cat(y_true)
    y_lag1 = torch.cat(y_lag1).squeeze().cpu().detach()
    
    
    train_data = train_dataloader.dataset[:][0][:,:,burn_in_steps,:]
    train_data = torch.cat([train_dataloader.dataset[0][0][:,:30,:].permute(1,0,2), train_data])
    train_data_ha = train_data.mean(dim=0)
    y_ha = train_data_ha.unsqueeze(dim=0).repeat(y_lag1.shape[0],1,1)
    
    return y_lag1, y_ha, y_true
