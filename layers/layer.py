import torch
import torch.nn as nn
import torch.nn.functional as F
CUDA = torch.cuda.is_available()


class ConvKBLayer(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super(ConvKBLayer, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels, (1, input_seq_len))
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear(input_dim * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))
        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output

class OnlyETProductLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_ent, drop_rate=0.4):
        super(OnlyETProductLayer, self).__init__()
        self.w_rel = nn.Parameter(torch.rand(in_channels, out_channels))
        nn.init.xavier_uniform_(self.w_rel.data)
        self.w_e_in = nn.Parameter(torch.rand(in_channels, out_channels, 3))
        nn.init.xavier_uniform_(self.w_e_in.data)
        self.w_e_out = nn.Parameter(torch.rand(in_channels, out_channels, 3))
        nn.init.xavier_uniform_(self.w_e_out.data)
        self.loop_e_w = nn.Parameter(torch.rand(in_channels, out_channels))
        nn.init.xavier_uniform_(self.loop_e_w.data)

        self.drop = nn.Dropout(drop_rate)
        self.bn_e = torch.nn.BatchNorm1d(out_channels)
        self.act = torch.tanh

    def tt_product(self, x_new, w_new):
        h = torch.zeros([x_new.shape[0], w_new.shape[1], x_new.shape[2]]).cuda()
        h[:, :, 0] = torch.matmul(x_new[:, :, 0], w_new[:, :, 0])+torch.matmul(x_new[:, :, 2], w_new[:, :, 2])+torch.matmul(x_new[:, :, 1], w_new[:, :, 1])
        h[:, :, 1] = torch.matmul(x_new[:, :, 1], w_new[:, :, 0])+torch.matmul(x_new[:, :, 0], w_new[:, :, 2])+torch.matmul(x_new[:, :, 2], w_new[:, :, 1])
        h[:, :, 2] = torch.matmul(x_new[:, :, 2], w_new[:, :, 0])+torch.matmul(x_new[:, :, 1], w_new[:, :, 2])+torch.matmul(x_new[:, :, 0], w_new[:, :, 1])
        return h


    def forward(self, all_e_embed, all_r_embed, init_x, subj, rel, case_study=False):
        if case_study == False:
            e_num = all_e_embed.shape[0]
            e_msg = torch.cat([self.tt_product(all_e_embed[:e_num // 2, :], self.w_e_in),
                         self.tt_product(all_e_embed[e_num // 2:, :], self.w_e_out)])

            h = self.drop(e_msg).cuda()
            all_multimodal_ent = self.act(self.bn_e(h))
            all_init_ent = torch.matmul(init_x, self.loop_e_w)
            all_ent_multimodal = torch.sum(all_multimodal_ent, dim=2)
            all_ent_multimodal = torch.nn.functional.softmax(all_ent_multimodal, dim=1)
            # all_ent = all_ent_multimodal + all_init_ent
            all_ent = all_ent_multimodal
            ent_emb = all_ent[subj]

            all_r = torch.matmul(all_r_embed, self.w_rel)
            rel_emb = torch.index_select(all_r, 0, rel)
        else:
            e_num = all_e_embed.shape[0]
            e_msg = torch.cat([self.tt_product(all_e_embed[:e_num // 2, :], self.w_e_in),
                         self.tt_product(all_e_embed[e_num // 2:, :], self.w_e_out)])
            h = self.drop(e_msg).cuda()
            all_multimodal_ent = self.act(self.bn_e(h))
            all_init_ent = torch.matmul(init_x, self.loop_e_w)
            all_ent_multimodal = torch.sum(all_multimodal_ent, dim=2)
            all_ent_multimodal = torch.nn.functional.softmax(all_ent_multimodal, dim=1)
            # all_ent = all_ent_multimodal + all_init_ent
            all_ent = all_ent_multimodal
            ent_emb = all_ent[subj]

            all_r = torch.matmul(all_r_embed, self.w_rel)
            rel_emb = 0

        return all_ent, ent_emb, rel_emb, all_multimodal_ent, all_init_ent, all_r

class InGramRelationLayer(nn.Module):
    def __init__(self, dim_in_rel, dim_out_rel, num_bin, bias=True, num_head=8):
        super(InGramRelationLayer, self).__init__()

        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head
        self.attn_drc = nn.Parameter(torch.zeros(4, num_head, 1))
        self.attn_proj = nn.Linear(2 * dim_in_rel, dim_out_rel, bias=bias)
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias=bias)
        self.num_head = num_head

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.num_bin = num_bin
        self.bias = bias

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_drc, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)

    def forward(self, emb_rel, relation_triplets):
        num_rel = len(emb_rel)

        head_idxs = relation_triplets[..., 0]
        tail_idxs = relation_triplets[..., 1]
        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim=-1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) * self.attn_vec).sum(dim=-1, keepdim=True) + self.attn_bin[
                           relation_triplets[..., 2]]  # + self.attn_drc[relation_triplets[...,3]]

        scatter_idx = head_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)

        attn_val_max = torch.zeros((num_rel, self.num_head, 1)).cuda().scatter_reduce(dim=0, index=scatter_idx, src=attn_val_raw, reduce='amax', include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])

        attn_sums = torch.zeros((num_rel, self.num_head, 1)).cuda().index_add(dim=0, index=head_idxs, source=attn_val)

        beta = attn_val / (attn_sums[head_idxs] + 1e-16)

        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel)).cuda().index_add(dim=0, \
                                                                                          index=head_idxs,
                                                                                          source=beta * self.aggr_proj(
                                                                                              emb_rel[tail_idxs]).view(
                                                                                              -1, self.num_head,
                                                                                              self.dim_hid_rel))

        return output.flatten(1, -1)

class ConvELayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_ent):
        super(ConvELayer, self).__init__()


        self.input_drop, self.conve_hid_drop, self.feat_drop = 0.4, 0.5, 0.5
        self.num_filt = 16
        self.ker_sz, self.k_w, self.k_h = 9, 32, 8


        self.embed_dim = in_channels
        self.bias = False
        self.num_ent = num_ent
        self.act = torch.tanh
        self.bias_m = nn.Parameter(torch.zeros(self.num_ent))

        self.bn0 = torch.nn.BatchNorm2d(1)  # one channel, do bn on initial embedding
        self.bn1 = torch.nn.BatchNorm2d(self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        # self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(self.input_drop)  # stacked input dropout
        self.feature_drop = torch.nn.Dropout(self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(self.conve_hid_drop)  # hidden layer dropout

        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=self.bias)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)  # fully connected projection

    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)  # [256, 1, 200]
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_input

    def forward(self, ent_embed, rel_embed, entity_embeddings):
        stack_input = self.concat(ent_embed, rel_embed)  # [batch_size, 1, 2*k_h, k_w]
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, entity_embeddings.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias_m.expand_as(x)
        x = torch.sigmoid(x)
        return x

class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices
            if(CUDA):
                edge_sources = edge_sources.to('cuda:0')
            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None

class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(out_features, 2 * in_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge):
        N = input.size()[0]

        edge_h = torch.cat((input[edge[0, :], :], input[edge[1, :], :]), dim=1).t()
        # edge_h: (2*in_dim) x E

        edge_m = self.W.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_m).squeeze())).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(edge, edge_w, N, edge_w.shape[0], self.out_features)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention
        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(nfeat, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nheads * nhid,
                                             nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, entity_embeddings, relation_embeddings, edge_list):
        x = entity_embeddings
        x = torch.cat([att(x, edge_list) for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x_rel = relation_embeddings.mm(self.W)
        x = F.elu(self.out_att(x, edge_list))
        return x, x_rel



