import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import negative_sampling
from torch_scatter import scatter_mean

EPS = 1e-15


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleEncoder, self).__init__()
        self.mu = nn.Parameter(torch.randn(in_channels, out_channels), requires_grad=True)
        self.logvar = nn.Parameter(torch.randn(in_channels, out_channels), requires_grad=True)

    def forward(self, x, edge_index):
        return x.matmul(self.mu), x.matmul(self.logvar)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logvar = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, weights, sigmoid=True, psi=None):
        if type(z) is tuple:
            z, c = z
            weights = {"vz": weights * 1, "vc": weights * 1, "vcz": weights * 1, "vzc": weights * 1}
            c = (c[:, :, None] * psi[None, :, :]).sum(1)
            v_cz = (c[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
            v_zc = (z[edge_index[0]] * c[edge_index[1]]).sum(dim=1)

            return (
                           torch.sigmoid(weights["vzc"] * v_zc) +
                           torch.sigmoid(weights["vcz"] * v_cz)
                   ) / 2
        else:
            value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
            return torch.sigmoid(value) if sigmoid else value


class Model(VGAE):
    def __init__(self, x_dim, z_dim, num_communities, q_alpha, encoder, decoder):
        super(Model, self).__init__(encoder, decoder)
        # TODO think about initialization if needed
        self.psi = nn.Parameter(torch.randn(num_communities, z_dim), requires_grad=True)
        self.q_alpha = q_alpha

    def community_dists_probs(self, z, edge_index):
        dot_products = (self.psi[None, :, :] * z[:, None, :]).sum(dim=2)
        row, col = edge_index
        dot_products_avg_over_Ni = scatter_mean(src=dot_products[row], index=col, dim=0, dim_size=z.size(0))
        weighted_dot_products = self.q_alpha * dot_products + (1 - self.q_alpha) * dot_products_avg_over_Ni
        return Categorical(logits=dot_products), Categorical(logits=weighted_dot_products)

    def recon_loss(self, z, pos_edge_index):
        pos_w, neg_w = 1.0, 1.0
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, pos_w, sigmoid=True, psi=self.psi) + EPS).mean()
        neg_edge_index = negative_sampling(pos_edge_index, z[0].size(0) if type(z) is tuple else z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, neg_w, sigmoid=True, psi=self.psi) + EPS).mean()
        return pos_loss + neg_loss

    def test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        if self.training:
            raise Exception("Cannot test in training mode")
        z = self.encode(x, train_pos_edge_index)
        pos_y = z.new_ones(test_pos_edge_index.size(1))
        neg_y = z.new_zeros(test_neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, test_pos_edge_index, 1, sigmoid=True, psi=self.psi)
        neg_pred = self.decoder(z, test_neg_edge_index, 1, sigmoid=True, psi=self.psi)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        return y, pred
