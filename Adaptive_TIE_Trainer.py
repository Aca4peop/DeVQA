'''
code for intradataset evaluation
'''

import warnings
from argparse import ArgumentParser

from tqdm import trange, tqdm

warnings.filterwarnings('ignore', category=FutureWarning)
import os.path
import torch
from torch.optim import AdamW,SGD
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
from SoftLabel import Labeler
from models.vqa import SingleBrach,Adaptor
from datasets.DataSource import LiveVQC,CVD2014,KonVid,LIVEVQA
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
        feat_t = np.load(os.path.join(self.fpath, vname[:-4] + '_text.npy'))
        dmos=self.dmos[index]
        sample = {'fv': feat_v, 'ft':feat_t,'label': dmos}
        return sample


if __name__ == "__main__":
    parser = ArgumentParser(description="Train Adaptive TIE module")
    parser.add_argument("--database", type=str, help="Database name.")
    args = parser.parse_args()
    datasets = {'konvid-1k': KonVid, 'live-vqa': LIVEVQA, 'live-vqc': LiveVQC, 'cvd2014': CVD2014}
    namings = {'konvid-1k': 'KonVid', 'live-vqa': 'LIVEVQA', 'live-vqc': 'LiveVQC', 'cvd2014': 'CVD2014'}
    device = torch.device("cuda")
    # parameters
    videoset = datasets[str(args.database)]()
    ft_path = './cache/features/%s/' % namings[args.database]

    srccs=np.zeros((5,3))
    rmses=np.zeros((5,3))
    plccs=np.zeros((5,3))

    records={}
    for rounder in range(5):
        trains,tests=videoset.get_five_folds(rounder)
        train_set = MMFeat(trains, fpath=ft_path)
        test_set = MMFeat(tests, fpath=ft_path)

        dataloader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4,pin_memory=True)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        visualb=SingleBrach(640,320).cuda()
        textb=SingleBrach(640,320).cuda()
        visualb.load_state_dict(torch.load('./cache/model/%s_visual_%d.pth'%(namings[args.database],rounder)))
        textb.load_state_dict(torch.load('./cache/model/%s_text_%d.pth'%(namings[args.database],rounder)))
        visualb.eval()
        textb.eval()
        adaptor=Adaptor(640,320).cuda()
        optimizer = AdamW([{'params': adaptor.parameters()}], lr=3e-4)
        kl = nn.KLDivLoss(reduction="batchmean")
        mse = nn.MSELoss()
        sroccbest_q=0
        lab=Labeler(videoset.scores)
        srcc_q=0
        rmse_q=0
        for epoch in trange(500):

            # ------train-------------
            adaptor.train()
            wq = torch.Tensor([5, 4, 3, 2, 1]).unsqueeze(0).to(device)
            L = 0
            for idx, data in enumerate(dataloader):
                feat_v = data['fv'].to(device).float()
                feat_t = data['ft'].to(device).float()
                label = data['label']
                label = lab(label)
                label = torch.from_numpy(label).float()
                label = label.to(device)
                mos = (label * wq).sum(dim=-1)
                q_v,fv=visualb.inference(feat_v)
                q_v=F.softmax(q_v,dim=-1)
                q_t, ft = textb.inference(feat_t)
                q_t = F.softmax(q_t, dim=-1)
                q=adaptor(fv,ft,q_v,q_t)
                q=q+1e-6
                q = q / q.sum(dim=-1, keepdim=True)
                loss=kl(q.log(),label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            record={}
            with torch.no_grad():
                adaptor.eval()
                tar = np.array([0])
                pre_vis = np.array([0])
                pre_text = np.array([0])
                pre_tde = np.array([0])
                wq = torch.Tensor([5, 4, 3, 2, 1]).unsqueeze(0)
                for idx, data in enumerate(testloader):
                    feat_v = data['fv'].to(device).float()
                    feat_t = data['ft'].to(device).float()
                    label = data['label']
                    labll = lab(label)
                    labll = torch.from_numpy(labll)
                    label = (labll * wq).sum(dim=-1).numpy().flatten()

                    q_v, fv = visualb.inference(feat_v)
                    q_v = F.softmax(q_v, dim=-1)
                    q_t, ft = textb.inference(feat_t)
                    q_t = F.softmax(q_t, dim=-1)
                    q = adaptor(fv, ft, q_v, q_t)
                    q = q+1e-6
                    q = q / q.sum(dim=-1, keepdim=True)

                    q_vis = (q_v.to('cpu') * wq).sum(dim=-1).numpy().flatten()
                    q_text = (q_t.to('cpu') * wq).sum(dim=-1).numpy().flatten()
                    q_tde = (q.to('cpu') * wq).sum(dim=-1).numpy().flatten()

                    tar = np.hstack((tar, label))
                    pre_vis = np.hstack((pre_vis, q_vis))
                    pre_text = np.hstack((pre_text, q_text))
                    pre_tde = np.hstack((pre_tde, q_tde))

                srcc_vis, _ = stats.spearmanr(pre_vis[1:], tar[1:])
                rmse_vis = rmse(pre_vis[1:], tar[1:])
                plcc_vis, _ = stats.pearsonr(pre_vis[1:], tar[1:])

                srcc_text, _ = stats.spearmanr(pre_text[1:], tar[1:])
                rmse_text = rmse(pre_text[1:], tar[1:])
                plcc_text, _ = stats.pearsonr(pre_text[1:], tar[1:])

                srcc_tde, _ = stats.spearmanr(pre_tde[1:], tar[1:])
                rmse_tde = rmse(pre_tde[1:], tar[1:])
                plcc_tde, _ = stats.pearsonr(pre_tde[1:], tar[1:])

                tqdm.write('Fold %d , epoch %d , SRCC %.4f , PLCC %.4f, RMSE %.4f' % (
                rounder, epoch, srcc_tde, plcc_tde, rmse_tde))
                if srcc_tde > sroccbest_q:
                    sroccbest_q = srcc_tde
                    srccs[rounder,0] = srcc_vis
                    rmses[rounder,0] = rmse_vis
                    plccs[rounder,0] = plcc_vis

                    srccs[rounder, 1] = srcc_text
                    rmses[rounder, 1] = rmse_text
                    plccs[rounder, 1] = plcc_text

                    srccs[rounder, 2] = srcc_tde
                    rmses[rounder, 2] = rmse_tde
                    plccs[rounder, 2] = plcc_tde
                    # torch.save({'vis':visualb.state_dict(),'txt':textb.state_dict(),'adp':adaptor.state_dict()},'./model%d.pth'%rounder)

    print('-------------Summary----------------')
    print('       SRCC  | PLCC  | RMSE')
    print('Visual %.4f | %.4f | %.4f'%(np.mean(srccs[:,0]),np.mean(plccs[:,0]),np.mean(rmses[:,0])))
    print('Text   %.4f | %.4f | %.4f' % (np.mean(srccs[:, 1]), np.mean(plccs[:, 1]), np.mean(rmses[:, 1])))
    print('TDE    %.4f | %.4f | %.4f' % (np.mean(srccs[:, 2]), np.mean(plccs[:, 2]), np.mean(rmses[:, 2])))



