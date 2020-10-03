import warnings

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch_geometric.utils import *

from dataset_loaders import load_facebook_dataset, FacebookDataset, load_large_dataset, LargeDataset
from helpers import scores, Scores, kv_to_print_str, matrix_to_cnl_format
from model import Model, InnerProductDecoder, SimpleEncoder, Encoder

warnings.filterwarnings("ignore")

##################

dataset_name = LargeDataset.DBLP
dataset_name = FacebookDataset.EgoFacebook3437

channels = 16
allow_features = False
community_pred_threshold = 0.3

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################
if type(dataset_name) is FacebookDataset:
    data = load_facebook_dataset(dataset_name=dataset_name, allow_features=allow_features)
elif type(dataset_name) is LargeDataset:
    data = load_large_dataset(dataset_name=dataset_name)

num_communities = data.num_communities
communities_cnl_format = data.communities_cnl_format
nx_graph = to_networkx(data).to_undirected()

data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.to(dev), torch.cat(
    (data.train_pos_edge_index,
     data.test_pos_edge_index,
     data.val_pos_edge_index), 1).to(dev)

# encoder = Encoder(in_channels=data.x.size(1), out_channels=channels)  # for amazon/youtube/dblp
encoder = SimpleEncoder(in_channels=data.x.size(1), out_channels=channels)
decoder = InnerProductDecoder()

model = Model(x.size(1), channels, num_communities, 0.9, encoder=encoder, decoder=decoder).to(dev)
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

        _, qc_given_ZA = model.community_dists_probs(model.__mu__, train_pos_edge_index)
        pre_comm_scores_weighted = (qc_given_ZA.probs / qc_given_ZA.probs.max(dim=1, keepdim=True)[0])
        pre_comm_scores_weighted_thresholded = (pre_comm_scores_weighted > community_pred_threshold).detach().cpu().numpy()
        pre_cnl_format = matrix_to_cnl_format(pre_comm_scores_weighted_thresholded.T, num_communities)

        labelled_idx = [x for r in communities_cnl_format for x in r]
        for i in range(len(pre_cnl_format)):
            pre_cnl_format[i] = [x for x in pre_cnl_format[i] if x in labelled_idx]

        metrics = scores(
            [Scores.COMMUNITY_OVERLAPPING_F1, Scores.COMMUNITY_OVERLAPPING_JACCARD],
            print_down=False, match_labels=False, communities_cnl=communities_cnl_format, communities_cnl_pred=pre_cnl_format
        )

        print("Epoch: {}\t".format(epoch) + kv_to_print_str(metrics))
