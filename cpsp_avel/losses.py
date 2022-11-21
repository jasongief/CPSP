import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def AVPSLoss(av_simm, soft_label):
    """audio-visual pair similarity loss for fully supervised setting,
    please refer to Eq.(8, 9) in our paper.
    """
    # av_simm: [bs, 10]
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss


   
def segment_contrastive_loss(fusion, frame_corr_one_hot, segment_flag_pos, segment_flag_neg, t=0.6):
    num_event = torch.sum(frame_corr_one_hot, dim=-1, keepdim=True) # [bs, 1]
    all_bg_flag = (num_event != 0).to(torch.float32)
    num_bg = 10 - num_event
    fusion = F.normalize(fusion, dim=-1) # [bs, 10, 256]
    cos_simm = torch.bmm(fusion, fusion.permute(0, 2, 1)) # [bs, 10, 10]
    
    mask_simm_neg = cos_simm * segment_flag_neg
    mask_simm_neg /= t
    mask_simm_exp_neg = torch.exp(mask_simm_neg) * segment_flag_neg

    mask_simm_pos = cos_simm * segment_flag_pos
    mask_simm_pos /= t
    mask_simm_exp_pos = torch.exp(mask_simm_pos) * segment_flag_pos

    simm_pos = torch.sum(mask_simm_exp_pos, dim=1) # [bs, 10], column-summation
    avg_simm_all_negs = torch.sum(mask_simm_exp_neg, dim=1) / (num_bg + 1e-12) # [bs, 10]
    simm_pos_all_negs = simm_pos + avg_simm_all_negs
    temp_result = simm_pos / (simm_pos_all_negs + 1e-12)
    loss = torch.sum((-1) * torch.log(temp_result + 1e-12), dim=-1, keepdim=True) / (num_event + 1e-12) # [bs, 1]
    # pdb.set_trace()
    loss *= all_bg_flag
    loss = torch.sum(loss) / torch.sum(all_bg_flag)
    return loss 


def video_contrastive_loss(fusion, batch_class_labels, margin=0.2, neg_num=3):
    """loss_vpsa used in PSA_V of CPSP"""
    # fusion: [bs, 10, dim=256], batch_class_labels: [bs,]
    fusion = F.normalize(fusion, dim=-1)
    avg_fea = torch.mean(fusion, dim=1) # [bs, 256]
    bs = avg_fea.size(0)
    dist = torch.pow(avg_fea, 2).sum(dim=1, keepdim=True).expand(bs, bs)
    dist = dist + dist.t()
    dist.addmm_(1, -2, avg_fea, avg_fea.t())
    dist = dist.clamp(min=1e-12).sqrt()

    mask = batch_class_labels.expand(bs, bs).eq(batch_class_labels.expand(bs, bs).t()).float()

    INF = 1e12
    NEG_INF = (-1) * INF

    dist_ap = dist * mask
    dist_an = dist * (1 - mask)

    dist_ap += torch.ones_like(dist_ap) * NEG_INF * (1 - mask)
    dist_an += torch.ones_like(dist_an) * INF * mask

    topk_dist_ap, topk_ap_indices = dist_ap.topk(k=1, dim=1, largest=True, sorted=True)
    topk_dist_an, topk_an_indices = dist_an.topk(k=neg_num, dim=1, largest=False, sorted=True)

    avg_topk_dist_ap = topk_dist_ap.squeeze(-1)
    avg_topk_dist_an = topk_dist_an.mean(dim=-1)
    y = torch.ones_like(avg_topk_dist_an)

    loss = nn.MarginRankingLoss(margin=margin)(avg_topk_dist_an, avg_topk_dist_ap, y)
    return loss




def LabelFreeSelfSupervisedNCELoss(fea_a, fea_v, t=0.1):
    bs, seg_num, dim = fea_a.shape
    fea_a = F.normalize(fea_a, dim=-1) # [bs, 10, 256]
    fea_v = F.normalize(fea_v, dim=-1) # [bs, 10, 256]
    rs_fea_a = fea_a.reshape(-1, dim) # [bs*10, 256]
    rs_fea_v = fea_v.reshape(-1, dim) # [bs*10, 256]
    # pdb.set_trace()
    pos_av_simm = torch.sum(torch.mul(rs_fea_a, rs_fea_v), dim=-1) # [bs*10=N,]
    batch_simm = torch.mm(rs_fea_a, rs_fea_v.t()) # [N, N]
    
    pos_av_simm /= t
    batch_simm /= t
    pos_av_simm = torch.exp(pos_av_simm)
    batch_simm = torch.exp(batch_simm)
    
    pos1_neg_item = torch.sum(batch_simm, dim=-1) # [N,]
    loss1 = torch.mean((-1) * torch.log(pos_av_simm / pos1_neg_item))

    return loss1
        