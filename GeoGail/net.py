import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import math
import numpy as np


def gen_gaussian_dist(sigma=10):
    """Return a single-sided gaussian distribution weight array and its index.
    """
    u = 0
    x = np.linspace(0, 1, 100)
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / \
        (math.sqrt(2 * math.pi) * sigma)
    return x, y


class ATNetwork(nn.Module):
    """Basic Generator.
    """
    def __init__(
            self,
            total_locations=3025,
            time_scale=48,
            embedding_net=None,
            loc_embedding_dim=64,
            tim_embedding_dim=8,
            pre_pos_count_embedding_dim=8,
            stay_time_embedding_dim=8,
            hidden_dim=64,
            bidirectional=False,
            data='geolife',
            device=None,
            starting_sample='zero',
            starting_dist=None,
            return_prob=True):
        """

        :param total_locations:
        :param embedding_net:
        :param embedding_dim:
        :param hidden_dim:
        :param bidirectional:
        :param cuda:
        :param starting_sample:
        :param starting_dist:
        """
        super(ATNetwork, self).__init__()
        self.total_locations = total_locations
        self.time_scale = time_scale
        self.loc_embedding_dim = loc_embedding_dim
        self.tim_embedding_dim = tim_embedding_dim
        self.pre_pos_count_embedding_dim = pre_pos_count_embedding_dim
        self.stay_time_embedding_dim = stay_time_embedding_dim
        self.embedding_dim = loc_embedding_dim + tim_embedding_dim + stay_time_embedding_dim + pre_pos_count_embedding_dim 
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.linear_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.device = device
        self.data = data
        self.starting_sample = starting_sample
        self.return_prob = return_prob
        # process distance weights
        #self.M1 = np.load('./preprocess_data/%s/burst_matrix.npy' % self.data)
        #self.M2 = np.load('./preprocess_data/%s/distance_matrix.npy' % self.data)


        if self.starting_sample == 'real':
            self.starting_dist = torch.tensor(starting_dist).float()

        if embedding_net:
            self.embedding = embedding_net
        else:
            self.loc_embedding = nn.Embedding(
                num_embeddings=self.total_locations, embedding_dim=self.loc_embedding_dim, scale_grad_by_freq=False, sparse=False)
            self.tim_embedding = nn.Embedding(
                num_embeddings=self.time_scale, embedding_dim=self.tim_embedding_dim, scale_grad_by_freq=False, sparse=False)
            self.stay_time_embedding = nn.Embedding(
                num_embeddings=self.time_scale, embedding_dim=self.stay_time_embedding_dim, scale_grad_by_freq=False, sparse=False)
            self.pre_pos_count_embedding = nn.Embedding(
                num_embeddings=self.time_scale, embedding_dim=self.pre_pos_count_embedding_dim, scale_grad_by_freq=False, sparse=False)
            self.home_embedding = nn.Embedding(
                num_embeddings=self.time_scale, embedding_dim=self.pre_pos_count_embedding_dim, scale_grad_by_freq=False, sparse=False)

        self.attn = nn.MultiheadAttention(self.hidden_dim, 8)
        self.Q = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.V = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.K = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.W = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Y = nn.Linear(self.hidden_dim, 64)


        self.linear = nn.Linear(self.linear_dim, self.total_locations)
        if self.return_prob:
            self.output_layer = nn.Linear(self.total_locations, 4)
        else:
            self.output_layer = nn.Linear(self.total_locations, 1)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

    def init_hidden(self, batch_size):
        h = torch.LongTensor(torch.zeros(
            (2 if self.bidirectional else 1, batch_size, self.hidden_dim)))
        c = torch.LongTensor(torch.zeros(
            (2 if self.bidirectional else 1, batch_size, self.hidden_dim)))
        if self.device:
            h, c = h.to(self.device), c.to(self.device)
        return h, c

    def forward(self, x_l, x_t, pre_pos_count, stay_time):
        """
        :param x: (batch_size, seq_len), sequence of locations
        :return:
            (batch_size * seq_len, total_locations), prediction of next stage of all locations
        """
        locs = x_l.contiguous().view(-1).detach().cpu().numpy()
        #mat1 = self.M1[locs]
        #mat2 = self.M2[locs]
        #mat1 = torch.Tensor(mat1).to(self.device)
        #mat2 = torch.Tensor(mat2).to(self.device)
        
        lemb = self.loc_embedding(x_l)
        temb = self.tim_embedding(x_t)
        semb = self.stay_time_embedding(stay_time)
        pemb = self.pre_pos_count_embedding(pre_pos_count)
        #hemb = self.stay_time_embedding(x_h)
        x = torch.cat([lemb, temb, semb, pemb], dim=-1)

        Query = self.Q(x)
        Query = F.relu(Query)
        Query = Query.unsqueeze(0)

        Value = self.V(x)
        Value = F.relu(Value)
        Value = Value.unsqueeze(0)

        Key = self.K(x)
        Key = F.relu(Key)
        Key = Key.unsqueeze(0)

        x, _ = self.attn(Query, Key, Value)
        '''
        x = self.Q(x)
        x = F.relu(x)
        x = self.W(x)
        x = F.relu(x)
        x = self.W(x)
        x = F.relu(x)
        x = self.Y(x)
        '''
        pred = self.linear(
            x.contiguous().view(-1, self.linear_dim))

        #mat1 = torch.sigmoid(mat1)
        #mat2 = torch.sigmoid(mat2)
        pred = pred
        #pred = self.Y(pred)
        pred = self.output_layer(F.relu(pred))
        if self.return_prob:
            pred = F.softmax(pred, -1)
        return pred

    def act(self, x_l, x_t, pre_pos_count, stay_time):
        prob = self.forward(x_l, x_t, pre_pos_count, stay_time)
        print(prob)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()    
        #print(action.cpu().max())
        return action.cpu().item()

class Recover_CNN(nn.Module):
    """Basic discriminator.
    """

    def __init__(
            self,
            total_locations=3025,
            embedding_net=None,
            embedding_dim=64,
            dropout=0.6):
        super(Recover_CNN, self).__init__()
        num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        if embedding_net:
            self.embedding = embedding_net
        else:
            self.embedding = nn.Embedding(
                num_embeddings=total_locations,
                embedding_dim=embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, embedding_dim)) for (
            n, f) in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sum(num_filters), 2)
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.embedding(x).unsqueeze(
            1)  # batch_size * 1 * seq_len * emb_dim
        # [batch_size * num_filter * length]
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                 for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + \
            (1. - torch.sigmoid(highway)) * pred
        pred = F.log_softmax(self.linear(self.dropout(pred)), dim=-1)
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


class Discriminator(nn.Module):
    def __init__(
        self,
        total_locations=3025,
        time_scale=48,
        embedding_net=None,
        loc_embedding_dim=64,
        tim_embedding_dim=8,
        pre_pos_count_embedding_dim=8,
        stay_time_embedding_dim=8,
        act_embedding_dim=4,
        hidden_dim=64
    ):
        super(Discriminator, self).__init__()
        self.total_locations = total_locations
        self.time_scale = time_scale
        self.loc_embedding_dim = loc_embedding_dim
        self.tim_embedding_dim = tim_embedding_dim
        self.pre_pos_count_embedding_dim = pre_pos_count_embedding_dim
        self.stay_time_embedding_dim = stay_time_embedding_dim
        self.act_embedding_dim = act_embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = loc_embedding_dim + tim_embedding_dim + act_embedding_dim + pre_pos_count_embedding_dim + stay_time_embedding_dim 

        self.loc_embedding = nn.Embedding(
            num_embeddings=self.total_locations, embedding_dim=self.loc_embedding_dim)
        self.tim_embedding = nn.Embedding(
            num_embeddings=self.time_scale, embedding_dim=self.tim_embedding_dim)
        self.act_embedding = nn.Embedding(
            num_embeddings=4, embedding_dim=self.act_embedding_dim)
        self.stay_time_embedding = nn.Embedding(
                num_embeddings=self.time_scale, embedding_dim=self.stay_time_embedding_dim)
        self.pre_pos_count_embedding = nn.Embedding(
                num_embeddings=self.time_scale, embedding_dim=self.pre_pos_count_embedding_dim)



        self.mlp_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_l, x_t, x_a, pre_pos_count, stay_time):
        lemb = self.loc_embedding(x_l)
        temb = self.tim_embedding(x_t)
        aemb = self.act_embedding(x_a)
        semb = self.stay_time_embedding(stay_time)
        pemb = self.pre_pos_count_embedding(pre_pos_count)      


        x = torch.cat([lemb, temb, aemb, semb ,pemb], dim=-1)
        reward = self.mlp_layer(x)
        return reward
