import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        # print(cos)
        # print(self.mrg)
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss

# deep hashing with proxy loss and binarization loss
class Proxy_Anchor_with_label(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, class_embed=0, target_label=0):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        # 均值0,方差1的分布随机生成proxy，并且该proxy是可训练到最优化的。
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.class_embed = class_embed
        self.target_label = target_label
        self.mrg = mrg
        self.alpha = alpha
        self.m = 0.25

    # 以下是计算损失的位置，X是网络的输出特征【100,512】,T是标签【100】   P是proxy[100,512]batch_size=100
    def forward(self, X, T):
        P = self.proxies
        # cos--->[150,100][batch,nb_class]
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity，x*p.T
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)  # P_one_hot [150.100]
        N_one_hot = 1 - P_one_hot
        # 调节网络优化方向以及优化程度的超参数
        # 阈值参数
        delta_p = 1 - self.m
        delta_n = -1 + self.m
        # 优于最优值的边界
        op = 1 + self.m
        on = -1 - self.m
        # 调节反向传播的速度和大小
        alpha_p = op -cos
        alpha_n = cos - on
        pos_exp = torch.exp(alpha_p * (cos - delta_p))
        neg_exp = torch.exp(alpha_n * (cos + delta_n))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / (num_valid_proxies )
        neg_term = torch.log(1 + N_sim_sum).sum() / (self.nb_classes)
        loss = pos_term + neg_term

        # cross entropy loss
        crossentropyloss = nn.CrossEntropyLoss()
        crossentropyloss_output = crossentropyloss(self.class_embed, self.target_label)

        # binarization loss
        v1 = P_sim_sum - torch.sign(P_sim_sum)
        v2 = N_sim_sum - torch.sign(N_sim_sum)
        loss01 = torch.mean(torch.sum(v1 * v1))
        loss02 = torch.mean(torch.sum(v2 * v2))
        loss0 = (loss01 + loss02)/2
        return loss + loss0 + crossentropyloss_output
def cosh(x):
    return (torch.exp(x)+torch.exp(-1.0*x))/2.0

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss

class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss