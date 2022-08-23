import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pdb



class avsdn_net(nn.Module):
    def __init__(self, vis_fea_type='vgg', flag='fully', hidden_dim=128, hidden_size=512, nb_block=1, category_num=28, pooling_type='avg'):
        super(avsdn_net, self).__init__()
        if vis_fea_type == 'vgg':
            self.v_init_dim = 512
        else:
            self.v_init_dim = 1024
        self.flag = flag
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim
        self.lstm_audio = nn.LSTM(
            128, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.lstm_video = nn.LSTM(
            self.v_init_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.lstm_a_v = nn.LSTM(
            self.v_init_dim + 128, hidden_dim, 1, batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(128, hidden_size)  # v_i = W_a * A
        self.affine_video = nn.Linear(self.v_init_dim, hidden_size)  # v_g = W_b * a^g
        self.affine_v = nn.Linear(hidden_size, 49, bias=False)  # W_v
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)  # W_g
        self.affine_h = nn.Linear(49, 1, bias=False)  # w_h

        self.event_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, 1)
        )
        self.category_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, category_num)
        )


        self.U_v = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )
        self.U_a = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )

        self.init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform_(self.affine_v.weight)
        init.xavier_uniform_(self.affine_g.weight)
        init.xavier_uniform_(self.affine_h.weight)
        # init.xavier_uniform_(self.affine_s.weight)
        init.xavier_uniform_(self.category_classifier[0].weight)
        init.xavier_uniform_(self.event_classifier[0].weight)
        init.xavier_uniform_(self.affine_audio.weight)
        init.xavier_uniform_(self.affine_video.weight)


    def TBMRF_block(self, audio, video, nb_block):
        for i in range(nb_block):
            video_residual = video
            v = self.U_v(video)
            audio_residual = audio
            a = self.U_a(audio)
            merged = torch.mul(v + a, 0.5)

            a_trans = audio_residual
            v_trans = video_residual

            video = torch.tanh(a_trans + merged)
            audio = torch.tanh(v_trans + merged)

        fusion = torch.mul(video + audio, 0.5)  #

        return fusion

    def seq2seq(self, audio, video_ori, video_att):  # [bs, 10, 128], [bs, 10, 2048], [bs, 10, 2048]
        lstm_video, (h_v, c_v) = self.lstm_video(video_ori)
        lstm_audio, (h_a, c_a) = self.lstm_audio(audio)
        mix_h = self.TBMRF_block(F.relu(h_a), F.relu(h_v), 1)
        mix_c = self.TBMRF_block(F.relu(c_a), F.relu(c_v), 1)
        f_av = torch.cat((audio, video_att), -1)

        # f_av = F.relu(f_av)
        lstm_av, (h_av, c_av) = self.lstm_a_v(f_av, (mix_h, mix_c))

        return lstm_av, h_av


    def forward(self, video, audio):
        # print('audio', audio.shape) # [bs, 10, 128]
        # print('video', video.shape) # [bs, 10, 7, 7, 512]
        # pdb.set_trace()
        v_t = video.view(video.size(0) * video.size(1), -1, self.v_init_dim) #[bs*10, 49, 512/1024]
        V = v_t
        v_t = self.relu(self.affine_video(v_t)) #[bs*10, 49, 512]

        a_t = audio.view(-1, audio.size(-1)) # [bs*10, 128]
        a_t = self.relu(self.affine_audio(a_t)) # [bs*10, 512]
        content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2) # [bs*10, 49, 1]

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2) # [bs*10, 49]
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) #[bs*10, 1, 49]
 
        # Construct c_t: B x seq x hidden_size
        c_t = torch.bmm(alpha_t, V).view(-1, 512) #[bs*10, 1, 49] x [bs*10, 49, 512] = [bs*10, 1, 512/1024]

        video_t = c_t.view(video.size(0), -1, 512) # [bs, 10, 512/1024]
        fusion, _ = self.seq2seq(
            audio.view(len(video), 10, -1), video_t.view(len(video), 10, -1),
            video_t.view(len(video), 10, -1)) # [bs, 10, 128], [bs, 10, 512], [bs, 10, 512]
        fusion = self.relu(fusion)


        if self.flag == 'fully':
            avps = None
            event_logits = self.event_classifier(fusion).squeeze(-1) # [B, 10]
            avg_fea = fusion.mean(dim=1) # [B, 256]
            category_logits = self.category_classifier(avg_fea) # [B, 28]
            # return fusion, avps, event_logits, category_logits
            return event_logits, category_logits, avps, fusion
        elif self.flag == 'weakly':
            event_logits = self.event_classifier(fusion) # [B, 10, 1]
            if self.pooling_type == 'avg':
                video_fea = fusion.mean(1) # [B, 256]
            elif self.pooling_type == 'max':
                video_fea, _ = fusion.max(1) # [B, 256]
                
            category_logits = self.category_classifier(video_fea)[:, None, :] # [B, 1, 29]
            fused_logits = event_logits.sigmoid() * category_logits # [B, 10, 29]

            logits, _ = torch.max(fused_logits, dim=1) # [B, 29]
            predict_video_labels = F.softmax(logits, dim=-1) # [B, 29]
            # return fusion, avps, event_logits, category_logits
            return fusion, event_logits.squeeze(), category_logits.squeeze(), predict_video_labels


if __name__ == "__main__":
    B = 4
    audio = torch.randn(B, 10, 128)
    vis_fea_type = 'vgg'
    if vis_fea_type == 'vgg':
        video = torch.randn(B, 10, 512, 7, 7)
        category_num = 28
    else:
        video = torch.randn(B, 10, 1024, 7, 7)
        category_num = 141


    fully_model  = avsdn_net(vis_fea_type, flag='fully', category_num=category_num)
    weakly_model = avsdn_net(vis_fea_type, flag='weakly', category_num=category_num)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fully_model.to(device)
    weakly_model.to(device)
    audio = audio.to(device)
    video = video.to(device)


    f1, f2, f3, f4 = fully_model(video, audio)
    w1, w2, w3, w4 = weakly_model(video, audio)
    pdb.set_trace()
