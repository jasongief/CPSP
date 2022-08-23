import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import copy
import math
from .PSP.fully_model import LSTM_A_V, PSP
import pdb


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v


class HANLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # src_q / src_v:[16, 10, 512]
        src_q = src_q.permute(1, 0, 2) # [10, 16, 512]
        src_v = src_v.permute(1, 0, 2) # [10, 16, 512]

        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0] # [10, 16, 512]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0] # [10, 16, 512]
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q) # [10, 16, 512]
        return src_q.permute(1, 0, 2) # [16, 10, 512]



class CMTLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(CMTLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src_q, src_v, src_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout1(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q



class HAN_MMIL_Net(nn.Module):

    def __init__(self):
        super(HAN_MMIL_Net, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)
        self.fc_a =  nn.Linear(128, 512)
        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.audio_encoder = nn.TransformerEncoder \
            (nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)
        self.visual_encoder = nn.TransformerEncoder \
            (nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)
        self.cmt_encoder = Encoder(CMTLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)
        self.hat_encoder = Encoder(HANLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)

    def forward(self, audio, visual, visual_st):
        # audio: [bs=16, 10, 128]
        # visual: [bs, 80, 2048]
        # visual_st: [bs, 10, 512]
        x1 = self.fc_a(audio) # [16, 10, 512]

        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1) # [16, 80, 2048] fc -> [16, 80, 512] permute -> [16, 512, 80] unsqueeze -> [16, 512, 80, 1]
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1) # [16, 512, 80, 1] avg_pool2d -> [16, 512, 10, 1] squeeze(-1) -> [16, 512, 10] permute -> [16, 10, 512]
        vid_st = self.fc_st(visual_st) # [16, 10, 512]
        x2 = torch.cat((vid_s, vid_st), dim =-1) #[16, 10, 512+512]
        x2 = self.fc_fusion(x2) # [16, 10, 512]

        # HAN
        x1, x2 = self.hat_encoder(x1, x2) # [16, 10, 512], [16, 10, 512]

        # prediction
        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2) # [16, 10, 2, 512]
        frame_prob = torch.sigmoid(self.fc_prob(x)) # [16, 10, 2, 25]

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1) # [16, 10, 2, 25], temporal weight
        av_att = torch.softmax(self.fc_av_att(x), dim=2) # [16, 10, 2, 25], modality weight
        temporal_prob = (frame_att * frame_prob) # [16, 10, 2, 25]
        global_prob = (temporal_prob *  av_att).sum(dim=2).sum(dim=1) # [16, 25]

        a_prob = temporal_prob[:, :, 0, :].sum(dim=1) # [16, 10, 25]
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1) # [16, 10, 25]

        return global_prob, a_prob, v_prob, frame_prob




class PSP_MMIL_Net(nn.Module):

    def __init__(self, args):
        super(PSP_MMIL_Net, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)

        self.fc_a =  nn.Linear(128, 512)#TODO:
        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512) #TODO:

        self.lstm = LSTM_A_V(a_dim=512, v_dim=512, hidden_dim=256)
        self.psp = PSP(a_dim=512, v_dim=512, hidden_dim=512, out_dim=512)
        self.thr_val = args.threshold


    def forward(self, audio, visual, visual_st):
        # audio: [bs=16, 10, 128]
        # visual: [bs, 80, 2048]
        # visual_st: [bs, 10, 512]
        b, t, d = visual_st.size()
        x1 = self.fc_a(audio) # [16, 10, 512]

        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1) # [16, 80, 2048] fc -> [16, 80, 512] permute -> [16, 512, 80] unsqueeze -> [16, 512, 80, 1]
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1) # [16, 512, 80, 1] avg_pool2d -> [16, 512, 10, 1] squeeze(-1) -> [16, 512, 10] permute -> [16, 10, 512]
        vid_st = self.fc_st(visual_st) # [16, 10, 512]
        x2 = torch.cat((vid_s, vid_st), dim =-1) #[16, 10, 512+512]
        x2 = self.fc_fusion(x2) # [16, 10, 512]
        x_visual = x2

        x1, x2 = self.lstm(x1, x2) # [B, 10, 512]
        _, x1, x2 = self.psp(x1, x2, thr_val=self.thr_val) # [B, 10, 512]

        sims = 1

        xx1 = F.normalize(x_visual, p=2, dim=-1)        
        xx2 = F.normalize(x1, p=2, dim=-1)

        sims = xx2.bmm(xx1.permute(0, 2, 1)).squeeze(1) / 0.2
        sims = sims.reshape(-1, 10)

        mask = torch.zeros(b, 10)
        mask = mask.long()
        for i in range(10):
            mask[:, i] = i
        mask = mask.cuda()
        mask = mask.reshape(-1)

        # prediction
        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2) # [16, 10, 2, 256]
        frame_prob = torch.sigmoid(self.fc_prob(x)) # [16, 10, 2, 25]
        
        # # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1) # [16, 10, 2, 25], temporal weight
        av_att = torch.softmax(self.fc_av_att(x), dim=2) # [16, 10, 2, 25], modality weight
        temporal_prob = (frame_att * frame_prob) # [16, 10, 2, 25]
        global_prob = (temporal_prob *  av_att).sum(dim=2).sum(dim=1) # [16, 25]


        a_prob = temporal_prob[:, :, 0, :].sum(dim=1) # [16, 10, 25]
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1) # [16, 10, 25]

        return x, global_prob, a_prob, v_prob, frame_prob, sims, mask


