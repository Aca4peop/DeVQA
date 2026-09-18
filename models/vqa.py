# Structure of the CIQNet

from mamba_ssm import Mamba

import torch
import torch.nn as nn
import torch.nn.functional as F

def similarity(tensor_1, tensor_2):
    normalized_tensor_1 = F.normalize(tensor_1, p=2, dim=-1)
    normalized_tensor_2 = F.normalize(tensor_2, p=2, dim=-1)
    cosine_sim = torch.sum(normalized_tensor_1 * normalized_tensor_2, dim=-1, keepdim=True)
    return 1-torch.mean(cosine_sim)


class MambaBlock(nn.Module):
    def __init__(self, d_model=256, d_ff=512):
        super().__init__()
        self.mamba = Mamba(d_model=d_model,d_state=16,d_conv=4,expand=1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 残差连接 + LayerNorm
        x = x + self.mamba(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SingleBrach(nn.Module):
    def __init__(self, dim_in=640,dim_hid1=512):
        super(SingleBrach, self).__init__()
        self.temp=MambaBlock(dim_in,int(dim_in*1.5))
        # self.pool=AttentionPool1d(dim_in,num_heads=8,output_dim=dim_hid1)
        # self.mlp1=nn.Identity()
        self.mlp1=nn.Sequential(
            nn.Linear(dim_in, dim_hid1),
            nn.LeakyReLU(True),
            nn.Linear(dim_hid1,5)
        )
    def forward(self,x):
        B,V,T,D=x.shape
        x=x.view(B*V,T,D)
        x=self.temp(x)
        x=x.mean(dim=-2)
        q=self.mlp1(x).squeeze(-1)
        # x=self.pool(x)
        q=q.view(B,V,5)
        q=q.mean(dim=1)
        return q

    def inference(self,x):
        B, V, T, D = x.shape
        x = x.view(B * V, T, D)
        x = self.temp(x)
        x = x.mean(dim=-2)
        q = self.mlp1(x)
        # x=self.pool(x)
        q = q.view(B, V, 5)
        q = q.mean(dim=1)
        return q,x.view(B,V,-1).mean(dim=1)

class Adaptor(nn.Module):
    def __init__(self,dim_in=640,dim_hid1=640):
        super(Adaptor,self).__init__()
        self.proj1=nn.Linear(dim_in,dim_hid1)
        self.proj2=nn.Linear(dim_in,dim_hid1)

        # self.fc=nn.Linear(dim_hid1,5)
        self.fc=nn.Sequential(nn.Linear(dim_hid1*2,dim_hid1),nn.ReLU(),nn.Linear(dim_hid1,5))

    def forward(self,fx,fy,qx,qy):
        fx=self.proj1(fx)
        fy=self.proj2(fy)
        diff=torch.cat((fx-fy,fx*fy),dim=-1)
        w=self.fc(diff)
        w=F.sigmoid(w)
        q=qx-w*qy
        q=torch.clip(q,min=0)
        return q