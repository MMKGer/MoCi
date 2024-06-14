import pickle
from layers.layer import *
import torch.nn as nn



class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = args.device

    @staticmethod
    def format_metrics(metrics, split):
        return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

    @staticmethod
    def has_improved(m1, m2):
        return (m1["Mean Rank"] > m2["Mean Rank"]) or (m1["Mean Reciprocal Rank"] < m2["Mean Reciprocal Rank"])

    @staticmethod
    def init_metric_dict():
        return {"Hits@100": -1, "Hits@10": -1, "Hits@3": -1, "Hits@1": -1,
                "Mean Rank": 100000, "Mean Reciprocal Rank": -1}

class OnlyConvKB(BaseModel):
    def __init__(self, args):
        super(OnlyConvKB, self).__init__(args)
        self.entityEmbed = nn.Parameter(torch.randn(len(args.entity2id), args.dim * args.n_heads))
        self.relationEmbed = nn.Parameter(torch.randn(len(args.relation2id), args.dim * args.n_heads))
        self.ConvKB = ConvKBLayer(args.dim * args.n_heads, 3, 1, args.out_channels, args.dropout, args.alpha)

    def forward(self, batch_inputs):
        head = self.entityEmbed[batch_inputs[:, 0]]
        relation = self.relationEmbed[batch_inputs[:, 1]]
        tail = self.entityEmbed[batch_inputs[:, 2]]
        conv_input = torch.cat((head.unsqueeze(1), relation.unsqueeze(1), tail.unsqueeze(1)), dim=1)
        conv_out = self.ConvKB(conv_input)
        return conv_out

    def loss_func(self, output, target):
        return F.soft_margin_loss(output, target)


class IKRLConvKB(BaseModel):
    def __init__(self, args):
        super(IKRLConvKB, self).__init__(args)
        self.imgEmbed = nn.Linear(args.img.shape[1], args.dim * args.n_heads)
        self.relationImgEmbed = nn.Parameter(torch.randn(len(args.relation2id), args.dim * args.n_heads))
        self.entityEmbed = nn.Parameter(torch.randn(len(args.entity2id), args.dim * args.n_heads))
        self.relationEmbed = nn.Parameter(torch.randn(len(args.relation2id), args.dim * args.n_heads))
        self.ConvKB = ConvKBLayer(args.dim * args.n_heads, 2 * 3, 1, args.out_channels, args.dropout, args.alpha)
        self.img = args.img.to(self.device)

    def forward(self, batch_inputs):
        head = torch.cat((self.entityEmbed[batch_inputs[:, 0]].unsqueeze(1),
                          self.imgEmbed(self.img[batch_inputs[:, 0]]).unsqueeze(1)), dim=1)
        relation = torch.cat((self.relationEmbed[batch_inputs[:, 1]].unsqueeze(1),
                              self.relationImgEmbed[batch_inputs[:, 1]].unsqueeze(1)), dim=1)
        tail = torch.cat((self.entityEmbed[batch_inputs[:, 2]].unsqueeze(1),
                          self.imgEmbed(self.img[batch_inputs[:, 2]]).unsqueeze(1)), dim=1)
        conv_input = torch.cat((head, relation, tail), dim=1)
        conv_out = self.ConvKB(conv_input)
        return conv_out

    def loss_func(self, output, target):
        return F.soft_margin_loss(output, target)


class IKRL(BaseModel):
    def __init__(self, args):
        super(IKRL, self).__init__(args)
        self.neg_num = args.neg_num
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(args.img.to(self.device).view(-1, 64, 64))
        img = img.view(img.size(0), -1)
        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)

    def forward(self, batch_inputs):
        len_pos_triples = int(batch_inputs.shape[0] / (int(self.neg_num) + 1))
        pos_triples = batch_inputs[:len_pos_triples]
        neg_triples = batch_inputs[len_pos_triples:]
        pos_triples = pos_triples.repeat(int(self.neg_num), 1)

        relation_pos = self.relation_embeddings(pos_triples[:, 1])
        relation_neg = self.relation_embeddings(neg_triples[:, 1])

        head_s_pos = self.entity_embeddings(pos_triples[:, 0])
        tail_s_pos = self.entity_embeddings(pos_triples[:, 2])

        head_s_neg = self.entity_embeddings(neg_triples[:, 0])
        tail_s_neg = self.entity_embeddings(neg_triples[:, 2])

        head_i_pos = self.img_entity_embeddings(pos_triples[:, 0])
        tail_i_pos = self.img_entity_embeddings(pos_triples[:, 2])

        head_i_neg = self.img_entity_embeddings(neg_triples[:, 0])
        tail_i_neg = self.img_entity_embeddings(neg_triples[:, 2])

        energy_ss_pos = torch.norm(head_s_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_ss_neg = torch.norm(head_s_neg + relation_neg - tail_s_neg, p=1, dim=1)

        energy_si_pos = torch.norm(head_s_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_si_neg = torch.norm(head_s_neg + relation_neg - tail_i_neg, p=1, dim=1)

        energy_is_pos = torch.norm(head_i_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_is_neg = torch.norm(head_i_neg + relation_neg - tail_s_neg, p=1, dim=1)

        energy_ii_pos = torch.norm(head_i_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_ii_neg = torch.norm(head_i_neg + relation_neg - tail_i_neg, p=1, dim=1)

        energy_pos = energy_ss_pos + energy_si_pos + energy_is_pos + energy_ii_pos
        energy_neg = energy_ss_neg + energy_si_neg + energy_is_neg + energy_ii_neg

        y = -torch.ones(int(self.neg_num) * len_pos_triples).to(self.device)
        loss = F.margin_ranking_loss(energy_pos, energy_neg, y, margin=10.0)

        return loss

    def predict(self, batch_inputs):
        pos_triples = batch_inputs

        relation_pos = self.relation_embeddings(pos_triples[:, 1])

        head_s_pos = self.entity_embeddings(pos_triples[:, 0])
        tail_s_pos = self.entity_embeddings(pos_triples[:, 2])

        head_i_pos = self.img_entity_embeddings(pos_triples[:, 0])
        tail_i_pos = self.img_entity_embeddings(pos_triples[:, 2])

        energy_ss_pos = torch.norm(head_s_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_si_pos = torch.norm(head_s_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_is_pos = torch.norm(head_i_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_ii_pos = torch.norm(head_i_pos + relation_pos - tail_i_pos, p=1, dim=1)

        energy_pos = energy_ss_pos + energy_si_pos + energy_is_pos + energy_ii_pos

        return energy_pos


class MKGC(BaseModel):
    def __init__(self, args):
        super(MKGC, self).__init__(args)
        self.neg_num = args.neg_num
        self.entity_embeddings = nn.Embedding(len(args.entity2id), 2 * args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(len(args.relation2id), 2 * args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(args.img.to(self.device).view(-1, 64, 64))
        img = img.view(img.size(0), -1)

        txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
        txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
        txt = txt.view(txt.size(0), -1)

        self.img_entity_embeddings = nn.Embedding.from_pretrained(torch.cat([img, txt], dim=1), freeze=False)

    def forward(self, batch_inputs):
        len_pos_triples = int(batch_inputs.shape[0] / (int(self.neg_num) + 1))
        pos_triples = batch_inputs[:len_pos_triples]
        neg_triples = batch_inputs[len_pos_triples:]
        pos_triples = pos_triples.repeat(int(self.neg_num), 1)

        relation_pos = self.relation_embeddings(pos_triples[:, 1])
        relation_neg = self.relation_embeddings(neg_triples[:, 1])

        head_s_pos = self.entity_embeddings(pos_triples[:, 0])
        tail_s_pos = self.entity_embeddings(pos_triples[:, 2])

        head_s_neg = self.entity_embeddings(neg_triples[:, 0])
        tail_s_neg = self.entity_embeddings(neg_triples[:, 2])

        head_i_pos = self.img_entity_embeddings(pos_triples[:, 0])
        tail_i_pos = self.img_entity_embeddings(pos_triples[:, 2])

        head_i_neg = self.img_entity_embeddings(neg_triples[:, 0])
        tail_i_neg = self.img_entity_embeddings(neg_triples[:, 2])

        energy_ss_pos = torch.norm(head_s_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_ss_neg = torch.norm(head_s_neg + relation_neg - tail_s_neg, p=1, dim=1)

        energy_si_pos = torch.norm(head_s_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_si_neg = torch.norm(head_s_neg + relation_neg - tail_i_neg, p=1, dim=1)

        energy_is_pos = torch.norm(head_i_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_is_neg = torch.norm(head_i_neg + relation_neg - tail_s_neg, p=1, dim=1)

        energy_ii_pos = torch.norm(head_i_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_ii_neg = torch.norm(head_i_neg + relation_neg - tail_i_neg, p=1, dim=1)

        energy_sisi_pos = torch.norm((head_s_pos + head_i_pos) + relation_pos - (tail_s_pos + tail_i_pos), p=1, dim=1)
        energy_sisi_neg = torch.norm((head_s_neg + head_i_neg) + relation_neg - (tail_s_neg + tail_i_neg), p=1, dim=1)

        energy_pos = energy_ss_pos + energy_si_pos + energy_is_pos + energy_ii_pos + energy_sisi_pos
        energy_neg = energy_ss_neg + energy_si_neg + energy_is_neg + energy_ii_neg + energy_sisi_neg

        y = -torch.ones(int(self.neg_num) * len_pos_triples).to(self.device)
        loss = F.margin_ranking_loss(energy_pos, energy_neg, y, margin=10.0)

        return loss

    def predict(self, batch_inputs):
        pos_triples = batch_inputs

        relation_pos = self.relation_embeddings(pos_triples[:, 1])

        head_s_pos = self.entity_embeddings(pos_triples[:, 0])
        tail_s_pos = self.entity_embeddings(pos_triples[:, 2])

        head_i_pos = self.img_entity_embeddings(pos_triples[:, 0])
        tail_i_pos = self.img_entity_embeddings(pos_triples[:, 2])

        energy_ss_pos = torch.norm(head_s_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_si_pos = torch.norm(head_s_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_is_pos = torch.norm(head_i_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_ii_pos = torch.norm(head_i_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_sisi_pos = torch.norm((head_s_pos + head_i_pos) + relation_pos - (tail_s_pos + tail_i_pos), p=1, dim=1)

        energy_pos = energy_ss_pos + energy_si_pos + energy_is_pos + energy_ii_pos + energy_sisi_pos

        return energy_pos


class TuckER(BaseModel):
    def __init__(self, args):
        super(TuckER, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(),
                freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(
                    pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0),
                freeze=False)
        self.dim = args.dim
        self.TuckER = TuckERLayer(args.dim, args.r_dim)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs, lookup=None):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        pred = self.TuckER(e_embed, r_embed)
        if lookup is None:
            pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        else:
            pred = torch.bmm(pred.unsqueeze(1), self.entity_embeddings.weight[lookup].transpose(1, 2)).squeeze(1)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class ConvE(BaseModel):
    def __init__(self, args):
        super(ConvE, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        self.dim = args.dim
        self.k_w = args.k_w
        self.k_h = args.k_h
        self.ConvE = ConvELayer(args.dim, args.out_channels, args.kernel_size, args.k_h, args.k_w)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        e_embed = e_embed.view(-1, 1, self.dim)
        r_embed = r_embed.view(-1, 1, self.dim)
        embed = torch.cat([e_embed, r_embed], dim=1)
        embed = torch.transpose(embed, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))

        pred = self.ConvE(embed)
        pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class MoCi(BaseModel):
    def __init__(self, args):
        super(MoCi, self).__init__(args)
        self.num_ent = len(args.entity2id)
        self.num_rel = 2 * len(args.relation2id)
        self.entity_embeddings = nn.Embedding(self.num_ent, args.r_dim, padding_idx=None).to(self.device)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(self.num_rel, args.r_dim, padding_idx=None).to(self.device)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        if args.dataset == 'FB15K-237' or args.dataset == 'DB15K' or args.dataset == 'YAGO15K' or args.dataset == 'FB15K':
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
            self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None).to(
                self.device)
            nn.init.xavier_normal_(self.img_relation_embeddings.weight)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
            self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
            self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None).to(
                self.device)
            nn.init.xavier_normal_(self.txt_relation_embeddings.weight)
        elif args.dataset == 'VTKG-I' or args.dataset == 'VTKG-C' or args.dataset == 'WN18RR++':
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
            self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None).to(
                self.device)
            nn.init.xavier_normal_(self.img_relation_embeddings.weight)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
            self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
            self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None).to(
                self.device)
            nn.init.xavier_normal_(self.txt_relation_embeddings.weight)
        else:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 8, 1149))
            img = img.view(img.size(0), -1)
            self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
            self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None).to(self.device)
            nn.init.xavier_normal_(self.img_relation_embeddings.weight)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 6, 64))
            txt = txt.view(txt.size(0), -1)
            self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
            self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None).to(self.device)
            nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        self.dim = args.dim
        self.ConvE_S = ConvELayer(args.dim,  len(args.entity2id))
        self.ConvE_T = ConvELayer(args.dim,  len(args.entity2id))
        self.ConvE_V = ConvELayer(args.dim,  len(args.entity2id))
        self.ConvE_MM = ConvELayer(args.dim, len(args.entity2id))

        self.conve_s_ent_embed_feature_embed_r = torch.nn.Linear(args.dim * 2, args.dim)
        self.conve_t_ent_embed_feature_embed_r = torch.nn.Linear(args.dim * 2, args.dim)
        self.conve_v_ent_embed_feature_embed_r = torch.nn.Linear(args.dim * 2, args.dim)
        self.conve_mm_ent_embed_feature_embed_r = torch.nn.Linear(args.dim * 2, args.dim)

        self.OnlyETProductLayer = OnlyETProductLayer(args.dim, args.dim, len(args.entity2id))
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()


    def mul_r_contrastive_loss(self, e_embed_r, e_img_embed_r, e_txt_embed_r, e_mm_embed_r, pos_s_embed, pos_v_embed, pos_t_embed, pos_mm_embed,label=None):
        s_embed, v_embed, t_embed, mm_embed = e_embed_r / torch.norm(e_embed_r), e_img_embed_r / torch.norm(e_img_embed_r), e_txt_embed_r / torch.norm(
            e_txt_embed_r), e_mm_embed_r / torch.norm(e_mm_embed_r)
        pos_s_embed, pos_v_embed, pos_t_embed,pos_mm_embed = pos_s_embed / torch.norm(pos_s_embed), pos_v_embed / torch.norm(pos_v_embed), pos_t_embed / torch.norm(
            pos_t_embed), pos_mm_embed / torch.norm(pos_mm_embed)

        s_score = torch.mm(s_embed, pos_s_embed.t())
        v_score = torch.mm(v_embed, pos_v_embed.t())
        t_score = torch.mm(t_embed, pos_t_embed.t())
        m_score = torch.mm(mm_embed, pos_mm_embed.t())

        s_v_score = torch.mm(s_embed, pos_v_embed.t())
        s_t_score = torch.mm(s_embed, pos_t_embed.t())
        s_m_score = torch.mm(s_embed, pos_mm_embed.t())

        t_s_score = torch.mm(t_embed, pos_s_embed.t())
        t_v_score = torch.mm(t_embed, pos_v_embed.t())
        t_m_score = torch.mm(t_embed, pos_mm_embed.t())

        v_s_score = torch.mm(v_embed, pos_s_embed.t())
        v_t_score = torch.mm(v_embed, pos_t_embed.t())
        v_m_score = torch.mm(v_embed, pos_mm_embed.t())

        m_s_score = torch.mm(pos_mm_embed, pos_s_embed.t())
        m_t_score = torch.mm(pos_mm_embed, pos_t_embed.t())
        m_v_score = torch.mm(pos_mm_embed, v_embed.t())

        loss_contrastive = self.loss_contrastive(s_score, label)
        loss_contrastive += self.loss_contrastive(v_score, label)
        loss_contrastive += self.loss_contrastive(t_score, label)
        loss_contrastive += self.loss_contrastive(s_v_score, label)
        loss_contrastive += self.loss_contrastive(s_t_score, label)
        loss_contrastive += self.loss_contrastive(t_s_score, label)
        loss_contrastive += self.loss_contrastive(t_v_score, label)
        loss_contrastive += self.loss_contrastive(v_s_score, label)
        loss_contrastive += self.loss_contrastive(v_t_score, label)

        loss_contrastive += self.loss_contrastive(m_score, label)
        loss_contrastive += self.loss_contrastive(s_m_score, label)
        loss_contrastive += self.loss_contrastive(t_m_score, label)
        loss_contrastive += self.loss_contrastive(v_m_score, label)

        loss_contrastive += self.loss_contrastive(m_s_score, label)
        loss_contrastive += self.loss_contrastive(m_t_score, label)
        loss_contrastive += self.loss_contrastive(m_v_score, label)
        return loss_contrastive

    def loss_contrastive(self, dis, label, margin=1.0):
        loss_contrastive = torch.mean((1 - label) * torch.pow(dis, 2) +
                                      (label) * torch.pow(torch.clamp(margin - dis, min=0.0), 2))

        return loss_contrastive


    def forward(self, batch_inputs, train_values=None,  model_split='decoder'):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        e_img_embed = self.img_entity_embeddings(head)
        r_img_embed = self.img_relation_embeddings(relation)
        e_txt_embed = self.txt_entity_embeddings(head)
        r_txt_embed = self.txt_relation_embeddings(relation)
        all_mul_e_embed = torch.stack(
            (self.entity_embeddings.weight, self.txt_entity_embeddings.weight, self.img_entity_embeddings.weight),
            dim=2)
        tt_mm_entity_embeddings, e_tt_mm_embed, r_tt_mm_embed, all_multimodal_ent, all_init_ent, mm_relation_embeddings = self.OnlyETProductLayer(
            all_mul_e_embed,
            self.relation_embeddings.weight,
            self.entity_embeddings.weight,
            head, relation)

        e_s_embed_r = self.conve_s_ent_embed_feature_embed_r(torch.cat((e_embed, r_embed), dim=1))
        e_t_embed_r = self.conve_t_ent_embed_feature_embed_r(torch.cat((e_txt_embed, r_txt_embed), dim=1))
        e_v_embed_r = self.conve_v_ent_embed_feature_embed_r(torch.cat((e_img_embed, r_img_embed), dim=1))
        e_mm_embed_r = self.conve_mm_ent_embed_feature_embed_r(torch.cat((e_tt_mm_embed, r_tt_mm_embed), dim=1))

        if model_split == 'pre_train_decoder':
            tail = torch.max(train_values, dim=1).indices

            pos_s_embed = self.entity_embeddings(tail)
            pos_t_embed = self.txt_entity_embeddings(tail)
            pos_v_embed = self.img_entity_embeddings(tail)
            pos_mm_embed = tt_mm_entity_embeddings[tail]

            pos_s_embed = torch.cat((pos_s_embed, r_embed), dim=1)
            pos_t_embed = torch.cat((pos_t_embed, r_img_embed), dim=1)
            pos_v_embed = torch.cat((pos_v_embed, r_txt_embed), dim=1)
            e_mm_embed_r = torch.cat((e_txt_embed, e_mm_embed_r), dim=1)
            pred_s = self.ConvE_S(e_s_embed_r, r_embed, self.entity_embeddings.weight)
            return [pred_s, e_s_embed_r, e_v_embed_r, e_t_embed_r, e_mm_embed_r,
                    pos_s_embed, pos_v_embed, pos_t_embed, pos_mm_embed]
        else:
            pred_s = self.ConvE_S(e_s_embed_r, r_embed, self.entity_embeddings.weight)
            pred_t = self.ConvE_T(e_t_embed_r, r_txt_embed, self.txt_entity_embeddings.weight)
            pred_v = self.ConvE_V(e_v_embed_r, r_img_embed, self.img_entity_embeddings.weight)
            tt_pred_mm = self.ConvE_MM(e_mm_embed_r, r_tt_mm_embed, tt_mm_entity_embeddings)
            return [pred_s, pred_t, pred_v, tt_pred_mm]


    def loss_func(self, output, target, adj_target=None):
        loss_s = self.bceloss(output[0], target)
        loss_t = self.bceloss(output[1], target)
        loss_v = self.bceloss(output[2], target)
        loss_tt_mm = self.bceloss(output[3], target)
        return loss_s + loss_t + loss_v + loss_tt_mm


    def pre_loss_func(self, output, target, adj_target=None):
        loss_s = self.bceloss(output[0], target)
        label_e = torch.eye(output[1].shape[0]).cuda()
        mul_r_contrastive_loss= self.mul_r_contrastive_loss(output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8], label_e)
        return loss_s + mul_r_contrastive_loss


