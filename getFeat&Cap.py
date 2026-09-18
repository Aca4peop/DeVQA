import skvideo.io
import numpy as np
import torch
from models.clip_cap import Visual
from tqdm import tqdm
import os
os.environ['http_proxy']='10.108.11.53:7890'
os.environ['https_proxy']='10.108.11.53:7890'

def main(video_dir,feat_dir,vlength):
    i_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device='cuda').unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    i_std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device='cuda').unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    visual = Visual().to('cuda')

    videos = os.listdir(video_dir)
    # finished = os.listdir('./cache')
    videos = sorted(videos)

    for video in tqdm(videos):
        # if video[:-4] + '_text.npy' in finished:
        #     continue
        v_feat_visual = torch.Tensor()
        v_feat_text = torch.Tensor()
        v_feat_text2 = torch.Tensor()
        if 'ODV' in feat_dir and not ('ERP' in video or 'TSP' in video or 'RCMP' in video):
            continue

        ffmpeg = skvideo.io.FFmpegReader(video_dir + video,
                                         outputdict={'-vf': 'v360=e:c6x1', '-s':  '2304x384'})
        tic = 0
        for frame in ffmpeg.nextFrame():
            tic += 1
            if not tic % 3 == 0:
                continue
            blocks = np.split(frame, 6, axis=1)
            blocks = np.array(blocks, dtype='float32').transpose(0, 3, 1, 2)

            blocks=blocks[:,:,80:-80,80:-80]

            blocks = torch.from_numpy(blocks).to('cuda').div_(255.0).sub_(i_mean).div_(i_std)
            v_feat, cap, cap_beam = visual(blocks)
            t_feat = visual.text(cap).unsqueeze(1)
            v_feat_visual = torch.cat((v_feat_visual, v_feat.unsqueeze(1).to('cpu')), dim=1)
            v_feat_text = torch.cat((v_feat_text, t_feat.to('cpu')), dim=1)
            t_feat = visual.text(cap_beam).unsqueeze(1)
            v_feat_text2 = torch.cat((v_feat_text2, t_feat.to('cpu')), dim=1)

            if tic > vlength:
                break

            # gc.collect()
        v_feat_visual = v_feat_visual.numpy()
        v_feat_text = v_feat_text.numpy()
        v_feat_text2 = v_feat_text2.numpy()
        # np.save(feat_dir + video[:-4] + '_visual.npy', v_feat_visual)
        # np.save(feat_dir+ video[:-4] + '_text.npy', v_feat_text)
        # np.save(feat_dir + video[:-4] + '_text2.npy', v_feat_text2)
        ffmpeg.close()
        print(video)
        for c in cap:
            print(c)
        print('---------------------')
        for c in cap_beam:
            print(c)



if __name__ == "__main__":
    video_dirs=['/home2/ODV-half/','/home2/JVQD/Videos/','/home2/SVQD/']
    feat_dirs=['./feats384vit/ODV/','./feats384vit/JVQD/','./feats384vit/SVQD/']
    v_length=[300,120,500]
    for i in range(3):
        main(video_dirs[i],feat_dirs[i],v_length[i])


