# Structure of the CIQNet

# from mamba_ssm import Mamba
# from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleBrach(nn.Module):
    def __init__(self, dim_in=640,dim_hid1=640):
        super(SingleBrach, self).__init__()
        # self.pool=AttentionPool1d(dim_in,num_heads=8,output_dim=dim_hid1)
        # self.mlp1=nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(d_model=dim_in, nhead=8, dim_feedforward=2 * dim_hid1,
                                               dropout=0.2, batch_first=True,
                                               activation='gelu' )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        self.mlp1=nn.Sequential(
            nn.Linear(dim_in, dim_hid1),
            nn.LeakyReLU(True),
        )
        self.mlp2=nn.Sequential(
            nn.Linear(dim_hid1, dim_hid1),
            nn.LeakyReLU(True),
        )
        self.fc1 = nn.Linear(dim_hid1,5)

    def forward_features(self,x):
        # x = self.mlp1(x)
        x=self.encoder(x)
        # x=self.temp(x)
        # x=x.mean(dim=-2)
        # x=self.pool(x)
        x=self.mlp1(x)
        return x
    def forward(self,x):
        x=self.forward_features(x)
        q_x=self.fc1(x).mean(dim=-2)
        return q_x,x.mean(dim=-2)

class adaptor(nn.Module):
    def __init__(self,dim_in=640,dim_hid1=640,dim_in1=128):
        super(adaptor,self).__init__()
        self.proj1=nn.Linear(dim_in1,dim_hid1)
        self.proj2=nn.Linear(dim_in,dim_hid1)

        self.fc=nn.Sequential(nn.Linear(dim_hid1*2,dim_hid1),nn.ReLU(True),nn.Linear(dim_hid1,5))
    def forward(self,x,y):
        x=self.proj1(x)
        y=self.proj2(y)
        diff=x-y
        hm=x*y
        w=self.fc(torch.cat([diff,hm],dim=-1))
        # w=self.fc(diff)
        w=F.sigmoid(w)#.mean(1)
        return w

class VQAModel(nn.Module):
    def __init__(self,feat_dim=768,fv_dim=128):
        super().__init__()
        self.text=SingleBrach(feat_dim,384)
        self.adap=adaptor(feat_dim,128,fv_dim)

    def forward(self,f_vis,f_t,q_vis):
        q_text, f_text = self.text(f_t)
        q_text = F.softmax(q_text, dim=-1)
        q_vis=F.softmax(q_vis, dim=-1)
        w = self.adap(f_vis, f_text).unsqueeze(1)
        q_ = torch.clip(q_vis - w * q_text, min=1e-6)
        q_ = q_ / q_.sum(dim=-1, keepdim=True)
        return q_,q_text,q_vis

class VQAModel2(nn.Module):
    def __init__(self,feat_dim=768):
        super().__init__()
        self.text=SingleBrach(feat_dim,384)
        self.adap=adaptor(384,128,384)
        self.vis=SingleBrach(feat_dim,384)

    def forward(self,f_vis,f_t):
        q_text, f_text = self.text(f_t)
        q_text = F.softmax(q_text, dim=-1)
        q_visual, f_visual = self.vis(f_vis)
        q_vis=F.softmax(q_visual, dim=-1)
        w = self.adap(f_visual, f_text)
        q_ = torch.clip(q_vis - w * q_text, min=1e-6)
        q_ = q_ / q_.sum(dim=-1, keepdim=True)
        return q_,q_text,q_vis