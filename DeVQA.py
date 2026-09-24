
import sys
import warnings

import torch
import os
import numpy as np
import decord
decord.bridge.set_bridge('torch')
from decord import VideoReader,cpu
from models.clip_cap import Visual
from torchvision.transforms.v2 import CenterCrop,Resize,Normalize,Compose,FiveCrop
from models.vqa import SingleBrach,Adaptor
warnings.filterwarnings("ignore")
import torch.nn.functional as F


def get_features(video_path):
    assert os.path.exists(video_path)
    process = Compose([
        # Resize((720)),
        FiveCrop(288),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    process2 = Compose([
        Resize(288),
        CenterCrop(288),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    model = Visual().to('cuda')

    vr = VideoReader(video_path, ctx=cpu(0))
    indeces = np.linspace(0, len(vr) - 1, 30).astype(int)
    frames = vr.get_batch(indeces)

    frames = frames.permute(0, 3, 1, 2).to('cuda').to(torch.float16).div_(255.0)
    minHW = min(frames.shape[-1], frames.shape[-2])
    if minHW < 288:
        frame = process2(frames)
        frames = frame.unsqueeze(0).repeat(5, 1, 1, 1, 1)
    else:
        frames = process(frames)

    visual1 = torch.Tensor()
    text1 = torch.Tensor()
    for i in range(5):
        with torch.inference_mode():
            vision_output, captions = model(frames[i])
            visual1 = torch.cat([visual1, vision_output.cpu().unsqueeze(0)], dim=0)
            t_feat = model.text(captions)
            text1 = torch.cat([text1, t_feat.cpu().unsqueeze(0)], dim=0)
    return visual1,text1


def inference(video_path):
    feat_v,feat_t = get_features(video_path)
    device = torch.device("cuda")
    visualb = SingleBrach(640, 320).to(device)
    textb = SingleBrach(640, 320).to(device)
    adaptor = Adaptor(640, 320).to(device)
    weights=torch.load('./model.pth')
    visualb.load_state_dict(weights['vis'])
    textb.load_state_dict(weights['txt'])
    adaptor.load_state_dict(weights['adp'])
    visualb.eval()
    textb.eval()
    adaptor.eval()
    q_v, fv = visualb.inference(feat_v.unsqueeze(0))
    q_v = F.softmax(q_v, dim=-1)
    q_t, ft = textb.inference(feat_t.unsqueeze(0))
    q_t = F.softmax(q_t, dim=-1)
    q = adaptor(fv, ft, q_v, q_t)
    q = q + 1e-6
    q = q / q.sum(dim=-1, keepdim=True)
    q=q.mean()
    return 1 - q


if __name__ == "__main__":
    video_file = sys.argv[1]
    score = inference(video_file)
    print('quality prediction: %.6f' % score)


