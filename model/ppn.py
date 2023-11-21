import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention1(nn.Module):
    def __init__(self, input_dim, proto_dim, hidden_dim, drop_rate=0.4):
        super(Attention1, self).__init__()

        self.input_dim = input_dim
        self.proto_dim = proto_dim
        self.hidden_dim = hidden_dim

        self.W_q = nn.Linear(proto_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, proto_dim)
        self.W_v = nn.Linear(input_dim, proto_dim)

        self.dropout = nn.Dropout(p=drop_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, distance, h_t):  # b, p, d; b, d, h
        batch_size = distance.size(0)

        # input_q = self.W_q(torch.mean(distance, 2))  # b, h
        input_q = self.W_q(distance)  # b, h
        # b, d, h -> b, h, d -> b, h, p -> b, p, h
        input_k = self.W_k(h_t.transpose(-1, -2)).transpose(-1, -2)
        # b, d, h -> b, h, d -> b, h, p -> b, p, h
        input_v = self.W_v(h_t.transpose(-1, -2)).transpose(-1, -2)

        q = torch.reshape(
            input_q, (batch_size, self.hidden_dim, 1))  # b, h, 1
        e = torch.matmul(input_k, q).squeeze(2)  # b, p

        a = self.softmax(e)  # b, d
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze(1)  # b, h
        return v, a


class PPN(nn.Module):
    def __init__(self,
                 input_dim=17,
                 hidden_dim=32,
                 output_dim=1,
                 demo_dim=4,
                 num_prototypes=8,
                 drop_rate=0.4):
        super(PPN, self).__init__()

# hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # d_model
        self.output_dim = output_dim

        self.num_prototypes = num_prototypes
        self.prototype_shape = (self.num_prototypes,
                                self.input_dim+1, self.hidden_dim)
        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True)

        # layers
        self.GRUs = clones(nn.GRU(1, self.hidden_dim, batch_first=True),
                           self.input_dim)
        self.Attention1 = Attention1(
            self.input_dim+1, self.num_prototypes, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.demo_proj = nn.Linear(demo_dim, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.sim = nn.CosineSimilarity(dim=-1)

        self.layernorm = nn.LayerNorm((self.input_dim+1, self.hidden_dim))

    def forward(self, x=None, lens=None, static=None, train_proto=False):
        if train_proto:
            batch_size = self.num_prototypes
            h_t = self.prototype_vectors
            distance = self.attention_similarity(h_t, batch_size)  # b, d, p
            distance = self.dropout(distance)
            h_t1, attn1 = self.Attention1(distance, h_t)  # b, h
            logits = self.output(h_t1)
            logits = self.sigmoid(logits)
            return logits, None, None

        batch_size = x.size(0)
        feature_dim = x.size(2)
        assert (feature_dim == self.input_dim)

        if static is not None:
            static = self.demo_proj(static).unsqueeze(1)  # b, 1, h
        hs = []
        if lens is None:
            for i in range(feature_dim):
                _, h_ti = self.GRUs[i](x[:, :, i].unsqueeze(-1))
                hs.append(h_ti.squeeze())
        else:
            for i in range(feature_dim):
                _, h_ti = self.GRUs[i](pack_padded_sequence(
                    x[:, :, i].unsqueeze(-1), lens, batch_first=True))
                hs.append(h_ti.squeeze())
        h_t = torch.stack(hs, dim=1)

        if static is not None:
            h_t = torch.cat((h_t, static), dim=1)
        h_t = self.layernorm(h_t)

        distance = self.attention_similarity(h_t, batch_size)  # b, d, p
        distance_drop = self.dropout(distance)
        h_t1, attn1 = self.Attention1(distance_drop, h_t)  # b, h
        logits = self.output(h_t1)
        logits = self.sigmoid(logits)
        return logits, distance, h_t

    def push_forward(self, x, lens=None, static=None):
        '''this method is needed for the pushing operation'''
        feature_dim = x.size(2)
        if static is not None:
            static = self.demo_proj(static).unsqueeze(1)  # b, 1, h
        hs = []
        if lens is None:
            for i in range(feature_dim):
                _, h_ti = self.GRUs[i](x[:, :, i].unsqueeze(-1))
                hs.append(h_ti.squeeze())
        else:
            for i in range(feature_dim):
                _, h_ti = self.GRUs[i](pack_padded_sequence(
                    x[:, :, i].unsqueeze(-1), lens, batch_first=True))
                hs.append(h_ti.squeeze())
        h_t = torch.stack(hs, dim=1)
        if static is not None:
            h_t = torch.cat((h_t, static), dim=1)
        h_t = self.layernorm(h_t)
        return h_t.cpu()

    def attention_similarity(self, x, batch_size):
        x1 = x.unsqueeze(1).repeat(1, self.num_prototypes, 1, 1).reshape(batch_size, self.num_prototypes, (self.input_dim+1)*self.hidden_dim)  # b, p, d, h
        x2 = self.prototype_vectors.unsqueeze(0).repeat(batch_size, 1, 1,
                                                        1).reshape(batch_size, self.num_prototypes, (self.input_dim+1)*self.hidden_dim)  # b, p, d, h
        distance = self.sim(x1, x2)
        return distance

    def freeze(self):
        self.prototype_vectors.requires_grad_(False)
        self.GRUs.requires_grad_(False)
        self.demo_proj.requires_grad_(False)
        self.layernorm.requires_grad_(False)

    def unfreeze(self):
        self.prototype_vectors.requires_grad_(True)
        self.GRUs.requires_grad_(True)
        self.demo_proj.requires_grad_(True)
        self.layernorm.requires_grad_(True)
        