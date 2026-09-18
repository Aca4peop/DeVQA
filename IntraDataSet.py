'''
code for intradataset evaluation
'''
import warnings

from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
warnings.filterwarnings('ignore', category=FutureWarning)

import os.path
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
from SoftLabel import Labeler
from models.vqa import SingleBrach,adaptor
from datasets.DataSource import LiveVQC
import torch.nn.functional as F

def rmse(target,predict):
    return np.sqrt(((predict - target) ** 2).mean())


class MMFeat(Dataset):
    def __init__(self, info,fvpath, ftpath,r):
        self.images = info['files']
        self.dmos = info['scores']
        self.fvpath=fvpath
        self.ftpath=ftpath
        self.r=r

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        vname = self.images[index]

        feat_v=np.load(os.path.join(self.fvpath,'VQC%d'%self.r,vname[:-4]+'_feat.npy'))
        feat_t=np.load(os.path.join(self.ftpath,vname[:-4]+'_text.npy'))
        q_vis=np.load(os.path.join(self.fvpath,'VQC%d'%self.r,vname[:-4]+'_output.npy'))
        indexes = np.linspace(0, feat_t.shape[0]-1,10).astype(int)
        feat_t=feat_t[indexes]
        # feat_t=feat_t[indexes]
        dmos=self.dmos[index]
        sample = {'feat_v': feat_v,'q_v':q_vis ,'feat_t': feat_t, 'label': dmos,'vname':vname[:-4]}
        return sample


if __name__ == "__main__":
    device = torch.device("cuda")
    # parameters
    videoset=LiveVQC()
    ft_path='/home/hzy/PycharmProjects/BLIP_Caption/blip2/vqc/'
    simplevqa_path='/home/hzy/PycharmProjects/SimpleVQA/cache/'
    fastervqa_path='/home/hzy/PycharmProjects/FAST-VQA-and-FasterVQA/cache/'
    vsfa_path='/home/hzy/PycharmProjects/VSFA-master/cache/'
    gstvqa_path='/home/hzy/PycharmProjects/GSTVQA/cache/'
    ssl_path='/home/hzy/PycharmProjects/SSRLVQA/cache/'
    conviqt_path='/home/hzy/PycharmProjects/CONVIQT/cache/'
    fv_path=ssl_path

    srccs=np.zeros((5,3))
    rmses=np.zeros((5,3))
    plccs=np.zeros((5,3))

    for rounder in range(5):
        trains,tests=videoset.get_five_folds(rounder)
        train_set = MMFeat(trains, fvpath=fv_path, ftpath=ft_path,r=rounder)
        test_set = MMFeat(tests, fvpath=fv_path, ftpath=ft_path,r=rounder)

        dataloader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4,pin_memory=True)
        testloader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

        model=adaptor(640,128,256).cuda()
        optimizer = AdamW([{'params': model.parameters()}], lr=3e-4)
        textmodel=SingleBrach(512).cuda()
        textmodel.load_state_dict(torch.load('./ckpt/VQC_BLIP2_%d.pt'%rounder))
        textmodel.eval()
        kl = nn.KLDivLoss(reduction="batchmean")
        mse = nn.MSELoss()
        sroccbest_q=0
        lab=Labeler(videoset.scores)
        srcc_q=0
        rmse_q=0
        for epoch in range(100):

            # ------train-------------
            model.train()
            wq = torch.Tensor([5, 4, 3, 2, 1]).unsqueeze(0).to(device)
            L = 0
            for idx, data in enumerate(dataloader):
                visual = data['feat_v'].to(device).float()
                text = data['feat_t'].to(device).float()
                q_vis=data['q_v'].to(device).float()
                label = data['label']
                label = lab(label)
                label = torch.from_numpy(label).float()
                label = label.to(device)
                mos = (label * wq).sum(dim=-1)

                q_txt,v_text=textmodel(text)
                w=model(visual,v_text)
                q_vis=F.softmax(q_vis,dim=-1)
                q_txt=F.softmax(q_txt,dim=-1)
                q_ = torch.clip(q_vis - w * q_txt, min=1e-8)
                q_ = q_ / q_.sum(dim=-1, keepdim=True)
                loss=kl(q_.log(),label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                model.eval()
                pre_q = np.array([0])
                tar = np.array([0])
                pre_vis = np.array([0])
                pre_txt = np.array([0])
                wq = torch.Tensor([5, 4, 3, 2, 1]).unsqueeze(0)
                for idx, data in enumerate(testloader):
                    visual = data['feat_v'].to(device).float()
                    text = data['feat_t'].to(device).float()
                    label = data['label']
                    q_vis = data['q_v'].to(device).float()
                    labll = lab(label)
                    labll = torch.from_numpy(labll)
                    label = (labll * wq).sum(dim=-1).numpy().flatten()

                    q_txt, v_text = textmodel(text)
                    w = model(visual, v_text)
                    q_vis = F.softmax(q_vis, dim=-1)
                    q_txt = F.softmax(q_txt, dim=-1)
                    q_ = torch.clip(q_vis - w * q_txt, min=1e-8)
                    q_ = q_ / q_.sum(dim=-1, keepdim=True)

                    q_ = (q_.to('cpu') * wq).sum(dim=-1).numpy().flatten()
                    q_text = (q_txt.to('cpu')* wq).sum(dim=-1).numpy().flatten()
                    q_vis = (q_vis.to('cpu') * wq).sum(dim=-1).numpy().flatten()

                    pre_q = np.hstack((pre_q, q_))
                    pre_txt=np.hstack((pre_txt,q_text))
                    tar = np.hstack((tar, label))
                    pre_vis = np.hstack((pre_vis, q_vis))

                rmse_txt = rmse(pre_txt[1:], tar[1:])
                rmse_q = rmse(pre_q[1:], tar[1:])
                srcc_txt, _ = stats.spearmanr(pre_txt[1:], tar[1:])
                srcc_q, _ = stats.spearmanr(pre_q[1:], tar[1:])
                srcc_vis, _ = stats.spearmanr(pre_vis[1:], tar[1:])
                rmse_vis = rmse(pre_vis[1:], tar[1:])
                plcc_q, _ = stats.pearsonr(pre_q[1:], tar[1:])
                plcc_txt, _ = stats.pearsonr(pre_txt[1:], tar[1:])
                plcc_vis, _ = stats.pearsonr(pre_vis[1:], tar[1:])

                if srcc_q > sroccbest_q:
                    sroccbest_q = srcc_q
                    srccs[rounder, 0] = srcc_q
                    rmses[rounder, 0] = rmse_q
                    srccs[rounder, 1] = srcc_txt
                    rmses[rounder, 1] = rmse_txt
                    srccs[rounder, 2] = srcc_vis
                    rmses[rounder, 2] = rmse_vis
                    plccs[rounder, 0] = plcc_q
                    plccs[rounder, 1] = plcc_txt
                    plccs[rounder, 2] = plcc_vis
                    print('Epoch:', epoch, 'SRCC_q:', srcc_q, 'SRCC_txt:', srcc_txt, 'RMSE_q:', rmse_q, 'RMSE_txt:', rmse_txt, 'SRCC_vis:', srcc_vis, 'RMSE_vis:', rmse_vis)

    print('-------------Summary----------------')
    print('SRCC q: %.4f vis %.4f text %.4f'%(np.mean(srccs[:,0]),np.mean(srccs[:,2]),np.mean(srccs[:,1])))
    print('PLCC q: %.4f vis %.4f text %.4f' % (np.mean(plccs[:, 0]), np.mean(plccs[:, 2]), np.mean(plccs[:, 1])))
    print('RMSE q: %.4f vis %.4f text %.4f'%(np.mean(rmses[:,0]),np.mean(rmses[:,2]),np.mean(rmses[:,1])))



