'''
code for intradataset evaluation
'''
import math
import pickle
import warnings
from matplotlib import pyplot as plt
from tqdm import trange

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
warnings.filterwarnings('ignore', category=FutureWarning)

import os.path
import torch
from torch.optim import AdamW,SGD
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
from SoftLabel import Labeler
from TDENetworkv5 import SingleBrach,adaptor
from datasets.DataSource import KonVid,LiveVQC,LIVEQACOM
import torch.nn.functional as F

def rmse(target,predict):
    return np.sqrt(((predict - target) ** 2).mean())


class MMFeat(Dataset):
    def __init__(self, info,fpath):
        self.images = info['files']
        self.dmos = info['scores']
        self.fpath=fpath

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        vname = self.images[index]
        feat_v=np.load(os.path.join(self.fpath,vname[:-4]+'_visual.npy'))
        dmos=self.dmos[index]
        sample = {'feat': feat_v, 'label': dmos}
        return sample


if __name__ == "__main__":
    device = torch.device("cuda")
    # parameters
    videoset=LIVEQACOM()
    ft_path='/home/hzy/PycharmProjects/BLIP_Caption/blip2/Qua/'

    srccs=np.zeros((5,))
    rmses=np.zeros((5,))
    plccs=np.zeros((5,))

    records={}
    for rounder in range(5):
        trains,tests=videoset.get_five_folds(rounder)
        train_set = MMFeat(trains, fpath=ft_path)
        test_set = MMFeat(tests, fpath=ft_path)

        dataloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4,pin_memory=True)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        model=SingleBrach(1408,384).cuda()
        optimizer = AdamW([{'params': model.parameters()}], lr=3e-4)
        kl = nn.KLDivLoss(reduction="batchmean")
        mse = nn.MSELoss()
        sroccbest_q=0
        lab=Labeler(videoset.scores)
        srcc_q=0
        rmse_q=0
        for epoch in trange(300):

            # ------train-------------
            model.train()
            wq = torch.Tensor([5, 4, 3, 2, 1]).unsqueeze(0).to(device)
            L = 0
            for idx, data in enumerate(dataloader):
                feat = data['feat'].to(device).float()
                label = data['label']
                label = lab(label)
                label = torch.from_numpy(label).float()
                label = label.to(device)
                mos = (label * wq).sum(dim=-1)
                q_vis=model(feat)
                q_vis=F.softmax(q_vis,dim=-1)
                loss=kl(q_vis.log(),label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            record={}
            with torch.no_grad():
                model.eval()
                tar = np.array([0])
                pre_vis = np.array([0])
                wq = torch.Tensor([5, 4, 3, 2, 1]).unsqueeze(0)
                for idx, data in enumerate(testloader):
                    feat = data['feat'].to(device).float()
                    label = data['label']
                    labll = lab(label)
                    labll = torch.from_numpy(labll)
                    label = (labll * wq).sum(dim=-1).numpy().flatten()

                    q_vis = model(feat)
                    q_vis = F.softmax(q_vis, dim=-1)
                    q_vis = (q_vis.to('cpu') * wq).sum(dim=-1).numpy().flatten()

                    tar = np.hstack((tar, label))
                    pre_vis = np.hstack((pre_vis, q_vis))

                srcc_vis, _ = stats.spearmanr(pre_vis[1:], tar[1:])
                rmse_vis = rmse(pre_vis[1:], tar[1:])
                plcc_vis, _ = stats.pearsonr(pre_vis[1:], tar[1:])

                if srcc_vis > sroccbest_q:
                    sroccbest_q = srcc_vis
                    srccs[rounder] = srcc_vis
                    rmses[rounder] = rmse_vis
                    plccs[rounder] = plcc_vis

    print('-------------Summary----------------')
    print('SRCC : %.4f PLCC %.4f RMSE %.4f'%(np.mean(srccs[:]),np.mean(srccs[:]),np.mean(srccs[:])))



