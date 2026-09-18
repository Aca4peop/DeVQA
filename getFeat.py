import os
from argparse import ArgumentParser

os.environ['http_proxy']='10.108.10.93:7890'
os.environ['https_proxy']='10.108.10.93:7890'
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
import numpy as np
from tqdm import tqdm
import decord
decord.bridge.set_bridge('torch')
from decord import VideoReader,cpu
from models.clip_cap import Visual
from torchvision.transforms.v2 import CenterCrop,Resize,Normalize,Compose,FiveCrop

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract visual and textual features")
    parser.add_argument("-i", type=str, help="Input path to the video directory.")
    parser.add_argument("--database", type=str, help="Database name.")
    args = parser.parse_args()
    namings = {'konvid-1k': 'KonVid', 'live-vqa': 'LIVEVQA', 'live-vqc': 'LiveVQC', 'cvd2014': 'CVD2014'}
    videos = os.listdir(args.i)
    process=Compose([
        # Resize((720)),
        FiveCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    process2=Compose([
        Resize(224),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    model = Visual().to('cuda')

    if not os.path.exists('./cache/features/%s/' % namings[args.database]):
        os.makedirs('./cache/features/%s/' % namings[args.database])
    for idx,video in tqdm(enumerate(videos)):
        tqdm.write('Processing '+video)
        try:
            vr = VideoReader(os.path.join(args.i,video), ctx=cpu(0))
        except RuntimeError:
            continue
        indeces = np.linspace(0, len(vr) - 1, 30).astype(int)
        frames = vr.get_batch(indeces)

        frames=frames.permute(0, 3, 1, 2).to('cuda').to(torch.float16).div_(255.0)
        minHW=min(frames.shape[-1],frames.shape[-2])
        if minHW<224:
            frame=process2(frames)
            frames=frame.unsqueeze(0).repeat(5,1,1,1,1)
        else:
            frames = process(frames)

        visual1=torch.Tensor()
        text1=torch.Tensor()
        for i in range(5):
            with torch.inference_mode():
                vision_output,captions = model(frames[i])
                visual1=torch.cat([visual1,vision_output.cpu().unsqueeze(0)],dim=0)
                t_feat = model.text(captions)
                text1=torch.cat([text1,t_feat.cpu().unsqueeze(0)],dim=0)

        np.save('./cache/features/%s/'%namings[args.database] + video[:-4] + '_text.npy', text1.numpy())
        np.save('./cache/features/%s/' % namings[args.database] + video[:-4] + '_visual.npy', visual1.numpy())

