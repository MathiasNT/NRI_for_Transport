import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Encoder_Visualizer(object):
    def __init__(
        self, encoder, rel_rec, rel_send, burn_in, burn_in_steps, split_len,
    ):
        super(Encoder_Visualizer, self).__init__()
        self.rel_rec = rel_rec
        self.rel_send = rel_send
        self.encoder = encoder
        self.burn_in = burn_in
        self.burn_in_steps = burn_in_steps
        self.split_len = split_len

    def infer_graphs(self, data, gumbel_temp):
        graph_list = []
        graph_probs = []
        self.encoder.eval()
        for _, data in enumerate(data):
            with torch.no_grad():
                data = data.unsqueeze(dim=0).cuda()
                logits = self.encoder(data, self.rel_rec, self.rel_send)
                edges = F.gumbel_softmax(logits, tau=gumbel_temp, hard=True)
                edge_probs = F.softmax(logits, dim=-1)
                graph_list.append(edges.cpu())
                graph_probs.append(edge_probs.cpu())
        return graph_list, graph_probs


def visualize_all_graph_adj(graph_list, rel_send, rel_rec):
    _, axs = plt.subplots(len(graph_list), 1, figsize=(50, 50))
    for k, graph in enumerate(graph_list):
        for j in range(1):
            adj_matrix = torch.zeros(132, 132)
            for i, row in enumerate(graph[j]):
                if row[1]:
                    send = rel_send[i].argmax().item()
                    rec = rel_rec[i].argmax().item()
                    adj_matrix[send, rec] = 1
            axs[k].imshow(adj_matrix)
            axs[k].set_title(f"{j}")


def visualize_mean_graph_adj(graph_list, rel_send, rel_rec):
    all_graphs = torch.stack(graph_list[:-1])
    mean_graph = all_graphs.mean(dim=(0, 1))
    mean_adj_matrix = torch.zeros(132, 132)
    for i, row in enumerate(mean_graph):
        if row[1]:
            send = rel_send[i].argmax().item()
            rec = rel_rec[i].argmax().item()
            mean_adj_matrix[send, rec] = 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mean_adj_matrix)
