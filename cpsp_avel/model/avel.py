import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init

import pdb

class TBMRF_Net(nn.Module):
    '''
    two-branch/dual muli-modal residual fusion
    '''
    def __init__(self, vis_fea_type='vgg', flag='fully', hidden_dim=128, hidden_size=512, nb_block=1, category_num=28, pooling_type='avg'):
        super(TBMRF_Net, self).__init__()
        if vis_fea_type == 'vgg':
            self.v_init_dim = 512
        else:
            self.v_init_dim = 1024
        self.flag = flag
        self.pooling_type = pooling_type

        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.lstm_audio = nn.LSTM(128, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.lstm_video = nn.LSTM(self.v_init_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.affine_audio = nn.Linear(128, hidden_size)  
        self.affine_video = nn.Linear(self.v_init_dim, hidden_size)  
        self.affine_v = nn.Linear(hidden_size, 49, bias=False) 
        self.affine_g = nn.Linear(hidden_size, 49, bias=False) 
        self.affine_h = nn.Linear(49, 1, bias=False)

        # fusion transformation functions
        self.nb_block = nb_block
        
        self.U_v  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        self.U_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )


        self.event_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, 1)
        )
        self.category_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, category_num)
        )

        # self.init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def TBMRF_block(self, audio, video, nb_block):

        for i in range(nb_block):
            video_residual = video
            v = self.U_v(video)
            audio_residual = audio
            a = self.U_a(audio)
            merged = torch.mul(v + a, 0.5) 

            a_trans = audio_residual
            v_trans = video_residual

            video = self.tanh(a_trans + merged)
            audio = self.tanh(v_trans + merged)

        fusion = torch.mul(video + audio, 0.5)#
        return fusion

    # def init_weights(self):
    #     """Initialize the weights."""
    #     init.xavier_uniform(self.L2.weight)

    def forward(self, video, audio):
        # audio: [B, 10, 128]
        # video: [B, 10, 512/1024, 7, 7]
        video = video.permute(0, 1, 3, 4, 2) # [B, 10, 7, 7, 512/1024]
        v_t = video.reshape(video.size(0) * video.size(1), -1, self.v_init_dim) # [B*10, 49, 512/1024]
        V = v_t # [B*10, 49, 512/1024]

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t)) # [B*10, 49, 512]
        a_t = audio.view(-1, audio.size(-1)) # [B*10, 128]
        a_t = self.relu(self.affine_audio(a_t)) # [B*10, 512]
        content_v = self.affine_v(v_t) \
                    + self.affine_g(a_t).unsqueeze(2) # [B*10, 49, 49] + [B*10, 49, 1]
        z_t = self.affine_h((F.tanh(content_v))).squeeze(2) # [B*10, 49]
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1))  # attention map, [B*10, 1, 49]
        c_t = torch.bmm(alpha_t, V).view(-1, self.v_init_dim) # [B*10, 512/1024]
        video_t = c_t.view(video.size(0), -1, self.v_init_dim) # attended visual features, [B, 10, 512/1024]

        # BiLSTM for Temporal modeling
        # hidden1 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)),
        #            autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)))
        # hidden2 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)),
        #            autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)))
        hidden1 = ((torch.zeros(2, audio.size(0), self.hidden_dim).double().cuda()),
                   (torch.zeros(2, audio.size(0), self.hidden_dim).double().cuda()))
        hidden2 = ((torch.zeros(2, audio.size(0), self.hidden_dim).double().cuda()),
                   (torch.zeros(2, audio.size(0), self.hidden_dim).double().cuda()))
        self.lstm_video.flatten_parameters()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(
            audio.view(len(audio), 10, -1), hidden1)
        lstm_video, hidden2 = self.lstm_video(
            video_t.view(len(video), 10, -1), hidden2)

        # Feature fusion and prediction
        fusion = self.TBMRF_block(lstm_audio, lstm_video, self.nb_block)
        fusion = nn.ReLU()(fusion)

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
            return fusion, event_logits.squeeze(), category_logits.squeeze(), predict_video_labels




if __name__ == "__main__":
    B = 4
    audio = torch.randn(B, 10, 128)
    vis_fea_type = 'resnet'
    if vis_fea_type == 'vgg':
        video = torch.randn(B, 10, 512, 7, 7)
    else:
        video = torch.randn(B, 10, 1024, 7, 7)


    fully_model  = TBMRF_Net(vis_fea_type, flag='fully')
    weakly_model = TBMRF_Net(vis_fea_type, flag='weakly')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fully_model.to(device)
    weakly_model.to(device)
    audio = audio.to(device)
    video = video.to(device)


    f1, f2, f3, f4 = fully_model(video, audio)
    w1, w2, w3, w4 = weakly_model(video, audio)
    pdb.set_trace()
