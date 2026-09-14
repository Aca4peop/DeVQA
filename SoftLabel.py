import numpy as np

def get_binary_probs(mos, min_mos=1.0, max_mos=5.0):
    eps = 1e-8
    probs = [0, 0, 0, 0, 0]
    for idx in range(1, len(probs)):
        mos_left = min_mos + (idx - 1) / 4 * (max_mos - min_mos) - eps
        mos_right = min_mos + idx / 4 * (max_mos - min_mos) + eps
        if mos > mos_left and mos <= mos_right:
            probs[idx - 1] = (mos_right - mos) / (mos_right - mos_left)
            probs[idx] = (mos - mos_left) / (mos_right - mos_left)
            break
    assert np.array((np.array(probs) == 0)).sum() == 3
    assert round(np.array(probs).sum(), 5) == 1
    probs = probs[::-1]  # should start with "excellent" & end with "bad"
    return probs

def get_hard_probs(mos, min_mos=1.0, max_mos=5.0):
    eps = 1e-8
    probs = [0, 0, 0, 0, 0]
    for idx in range(1, len(probs)):
        mos_left = min_mos + (idx - 1) / 4 * (max_mos - min_mos) - eps
        mos_right = min_mos + idx / 4 * (max_mos - min_mos) + eps
        if mos > mos_left and mos <= mos_right:
            probs[idx - 1] = (mos_right - mos) / (mos_right - mos_left)
            probs[idx] = (mos - mos_left) / (mos_right - mos_left)
            break
    assert np.array((np.array(probs) == 0)).sum() == 3
    assert round(np.array(probs).sum(), 5) == 1
    probs = probs[::-1]  # should start with "excellent" & end with "bad"
    return probs

class Labeler:
    def __init__(self,mos):
        self.max_mos=np.max(mos)
        self.min_mos=np.min(mos)

    def __call__(self, mos):
        if isinstance(mos,int):
            probs=get_binary_probs(mos,self.min_mos,self.max_mos)
            return probs
        else:
            probs=[]
            for m in mos:
                prob=get_binary_probs(m,self.min_mos,self.max_mos)
                probs.append(prob)
            return np.array(probs)
