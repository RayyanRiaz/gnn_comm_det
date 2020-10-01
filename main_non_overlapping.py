import glob
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch_geometric.utils import *

from dataset_loaders import load_non_overlapping_dataset, CitationFullDataset
from helpers import predict_node_classification, scores, Scores, kv_to_print_str
from model import Model, InnerProductDecoder, SimpleEncoder

warnings.filterwarnings("ignore")
print("Warnings Ignored")
##################

# dataset_name = PlanetoidDataset.Cora
dataset_name = CitationFullDataset.CoraML

channels = 128
cross_prod_decoder = True
##################


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = load_non_overlapping_dataset(dataset_name)
communities = data.y
communities = communities.type(torch.int64).cpu().numpy()
num_communities = len(set(communities))
# encoder = Encoder(in_channels=data.x.size(1), out_channels=channels)  # GCN Encoder
encoder = SimpleEncoder(in_channels=data.x.size(1), out_channels=channels)
decoder = InnerProductDecoder()
###############
splits = {}
split_file_paths = glob.glob("splits/{}{}_Manual/*.npy".format(
    dataset_name.value, "_full" if type(dataset_name) == CitationFullDataset else ""))
for fpath in split_file_paths:
    splits[fpath.split("/")[-1].split(".npy")[0]] = np.load(fpath).astype(int).squeeze()
train_mask = torch.cuda.BoolTensor(data.num_nodes).zero_()
train_mask[splits["X_train"]] = True
data.train_mask_for_classification = train_mask
###############

model = Model(data.x.size(1), channels, num_communities, q_alpha=0.9, encoder=encoder, decoder=decoder).to(dev)
data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.to(dev), torch.cat(
    (data.train_pos_edge_index,
     data.test_pos_edge_index,
     data.val_pos_edge_index), 1).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(train_type):
    if train_type == 0:  # simple VGAE trained
        model.eval()
        for param in model.parameters():
            param.requires_grad = True
        model.psi.requires_grad = False

        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        l_kl_z = 1.0 * model.kl_loss() / data.num_nodes
        l_recon = model.recon_loss(z, train_pos_edge_index)
        l_kl_c = 0
        loss = l_recon + l_kl_z
        loss.backward()
        optimizer.step()
    elif train_type == 1:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.psi.requires_grad = True

        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        pc_given_Z, qc_given_ZA = model.community_dists_probs(z, train_pos_edge_index)
        c = F.gumbel_softmax(qc_given_ZA.logits, tau=1, hard=True)
        l_kl_z = 1.0 * model.kl_loss() / data.num_nodes
        l_recon = model.recon_loss((z, c), train_pos_edge_index)
        l_kl_c = 1.0 * kl_divergence(qc_given_ZA, pc_given_Z).mean()
        loss = l_recon + l_kl_z
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad = True

        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        pc_given_Z, qc_given_ZA = model.community_dists_probs(z, train_pos_edge_index)
        c = F.gumbel_softmax(qc_given_ZA.logits, tau=1, hard=True)
        l_kl_z = 1.0 * model.kl_loss() / data.num_nodes
        l_kl_c = 1.0 * kl_divergence(qc_given_ZA, pc_given_Z).mean()
        l_recon = model.recon_loss((z, c), train_pos_edge_index)
        loss = l_recon + l_kl_z + l_kl_c
        loss.backward()
        optimizer.step()

    return l_recon, l_kl_z, l_kl_c


for epoch in range(1, 10001):
    if epoch % 50 == 0 and epoch < 800:
        print("pretraining... Epoch: {}".format(epoch))

    if epoch < 400:
        tt = 0
    elif epoch < 800:
        tt = 1
    else:
        tt = 2
    l_recon, l_kl_z, l_kl_c = train(tt)

    if epoch % 10 == 0 and epoch > 800:
        model.eval()

        with torch.no_grad():
            edges, edges_pred = model.test(x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)
            edges, edges_pred = edges.detach().cpu().numpy(), edges_pred.detach().cpu().numpy()

            _, qc_given_ZA = model.community_dists_probs(model.__mu__, train_pos_edge_index)
            communities_pred = torch.argmax(qc_given_ZA.probs, dim=1).detach().cpu().numpy()

            node_classes = communities[~data.train_mask_for_classification.cpu().numpy()]
            node_classes_pred = predict_node_classification(
                model.__mu__[data.train_mask_for_classification], data.y[data.train_mask_for_classification],
                model.__mu__[~data.train_mask_for_classification], max_iter=500)

            metrics = scores(keys=[
                Scores.COMMUNITY_NMI, Scores.COMMUNITY_ARI, Scores.NODE_CLASSIFICATION_F1_MACRO, Scores.NODE_CLASSIFICATION_F1_MICRO],
                match_labels=True, print_down=False, edges=edges, edges_pred=edges_pred, communities=communities,
                communities_pred=communities_pred, node_classes=node_classes, node_classes_pred=node_classes_pred)

            print("Epoch: {:4d}\tLR: {:.4f}\tLZ: {:.4f}\tLC: {:.4f}\t".format(epoch, l_recon, l_kl_z, l_kl_c) + kv_to_print_str(metrics))

            with open("log.txt", "a") as f:
                print("========{}========".format(dataset_name.value), file=f)
                print("Epoch {:4d}: ||".format(epoch) + kv_to_print_str(metrics), file=f)
