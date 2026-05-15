import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Actor cho negative sampling
class ActorModule(nn.Module):
    def __init__(self, emb_size, hidden_size=64, n_heads=2, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        
        self.user_proj = nn.Linear(emb_size, hidden_size)
        self.item_proj = nn.Linear(emb_size, hidden_size)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Score predictor
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        
    def forward(self, user_emb, cand_embs):
        # user_emb: [batch_size, emb_size]
        # cand_embs: [batch_size, n_negs, emb_size]
        
        batch_size, n_negs, emb_size = cand_embs.shape
        
        # Project embeddings
        u_proj = self.user_proj(user_emb)  # [batch_size, hidden_size]
        c_proj = self.item_proj(cand_embs)  # [batch_size, n_negs, hidden_size]
        
        # Attention mechanism
        u_query = u_proj.unsqueeze(1)  # [batch_size, 1, hidden_size]
        c_att, _ = self.attention(u_query, c_proj, c_proj)  # [batch_size, 1, hidden_size]
        c_att = c_att.squeeze(1)  # [batch_size, hidden_size]
        
        # Combine user and attended candidates
        u_expanded = u_proj.unsqueeze(1).expand(-1, n_negs, -1)  # [batch_size, n_negs, hidden_size]
        c_att_expanded = c_att.unsqueeze(1).expand(-1, n_negs, -1)  # [batch_size, n_negs, hidden_size]
        
        # Score computation
        combined = torch.cat([u_expanded, c_att_expanded], dim=-1)  # [batch_size, n_negs, hidden_size*2]
        scores = self.score_head(combined).squeeze(-1)  # [batch_size, n_negs]
           
        # Softmax distribution
        probs = F.softmax(scores, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        idx = torch.argmax(probs, dim=-1)
        
        log_prob = dist.log_prob(idx)
        entropy = dist.entropy()
        
        return idx, log_prob, entropy, scores

#Critic cho value estimation
class CriticModule(nn.Module):
    def __init__(self, emb_size, hidden_size=64, dropout=0.1):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, user_emb):
        return self.value_net(user_emb).squeeze(-1)

class NGCF(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(NGCF, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        self.K = args_config.K

        """
        *********************************************************
        Init the weight of user-item.   
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        
        # RL components - đơn giản hóa
        if self.pool == 'concat':
            pooled_emb_size = self.emb_size * (self.context_hops + 1)
        else:
            pooled_emb_size = self.emb_size
            
        self.actor = ActorModule(
            emb_size=pooled_emb_size,
            hidden_size=getattr(args_config, 'rl_hidden', 64),
            n_heads=getattr(args_config, 'rl_heads', 2),
            dropout=getattr(args_config, 'rl_dropout', 0.1)
        ).to(self.device)
        
        self.critic = CriticModule(
            emb_size=pooled_emb_size,
            hidden_size=getattr(args_config, 'rl_hidden', 64),
            dropout=getattr(args_config, 'rl_dropout', 0.1)
        ).to(self.device)
        
        # RL hyperparameters - đơn giản hóa
        self.rl_weight = getattr(args_config, 'rl_weight', 0.01)  # Giảm weight xuống
        self.entropy_coef = getattr(args_config, 'entropy_coef', 0.01)
        self.value_coef = getattr(args_config, 'value_coef', 0.1)
        
    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items,
                                                 self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] * (self.context_hops+1)
        for k in range(self.context_hops):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, self.K, -1)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]

        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_rl_loss(self, rewards, values, log_probs, entropies):
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute advantages
        advantages = rewards - values.detach()
        
        # Actor loss (policy gradient)
        actor_loss = -(advantages * log_probs).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values, rewards)
        
        # Entropy bonus
        entropy_loss = -entropies.mean()
        
                
        return actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def gcn(self, edge_dropout=True, mess_dropout=True):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.edge_dropout_rate,
                                    self.sparse_norm_adj._nnz()) if edge_dropout else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(self.context_hops):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                            + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            if mess_dropout:
                ego_embeddings = nn.Dropout(self.mess_dropout_rate)(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)  # [n_entity, n_hops+1, emb_size]
        return all_embeddings[:self.n_users, :], all_embeddings[self.n_users:, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(edge_dropout=False, mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        

        
        # RL-based negative sampling
        if self.pool != 'concat':
            s_e_pooled = self.pooling(s_e)
        else:
            s_e_pooled = s_e.view(batch_size, -1)
            
        # Get candidate embeddings
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]
        
        # Pool candidate embeddings
        if self.pool != 'concat':
            n_e_pooled = self.pooling(n_e.view(-1, n_e.shape[2], n_e.shape[3])).view(batch_size, self.n_negs, -1)
        else:
            n_e_pooled = n_e.view(batch_size, self.n_negs, -1)
        
        # Use actor to select negative samples
        selected_idx, log_prob, entropy, scores = self.actor(s_e_pooled, n_e_pooled)
        
        # Get selected negative embeddings
        selected_neg_emb = n_e[torch.arange(batch_size), selected_idx]
        
        # Compute baseline (critic value)
        baseline = self.critic(s_e_pooled)
        
        # Compute reward (similarity difference)
        with torch.no_grad():
            s_e_pooled_detach = s_e_pooled.detach()
            p_e_pooled = self.pooling(p_e) if self.pool != 'concat' else p_e.view(batch_size, -1)
            n_e_selected_pooled = self.pooling(selected_neg_emb) if self.pool != 'concat' else selected_neg_emb.view(batch_size, -1)
            
            pos_sim = torch.sum(s_e_pooled_detach * p_e_pooled, dim=-1)
            neg_sim = torch.sum(s_e_pooled_detach * n_e_selected_pooled, dim=-1)
            reward = pos_sim - neg_sim  # Higher is better
        
        # Store RL information
        self.rl_info = {
            'reward': reward,
            'baseline': baseline,
            'log_prob': log_prob,
            'entropy': entropy,
            'selected_idx': selected_idx
        }
        
        return selected_neg_emb

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def forward(self, batch):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        

        user_gcn_emb, item_gcn_emb = self.gcn(edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        pos_gcn_embs = item_gcn_emb[pos_item]

        neg_gcn_embs = []
        rl_losses = []
        
        for k in range(self.K):
            neg_emb = self.negative_sampling(user_gcn_emb, item_gcn_emb,
                                           user, neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                           pos_item)
            neg_gcn_embs.append(neg_emb)
            
            # Collect RL loss if available
            if self.rl_info is not None:
                rl_loss= self.create_rl_loss(
                    self.rl_info['reward'],
                    self.rl_info['baseline'],
                    self.rl_info['log_prob'],
                    self.rl_info['entropy']
                )
                rl_losses.append(rl_loss)

        neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)
        
        # Compute BPR loss
        bpr_loss, mf_loss, emb_loss = self.create_bpr_loss(user_gcn_emb[user], pos_gcn_embs, neg_gcn_embs)
        
        # Add RL loss if available

        rl_loss_avg = torch.stack(rl_losses).mean()
        
        total_loss = bpr_loss + self.rl_weight * rl_loss_avg
        
        return total_loss, mf_loss, emb_loss