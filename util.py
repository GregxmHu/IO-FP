import random
import torch
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def cal_distance(a):
    b=a.unsqueeze(0).expand((a.shape[0],a.shape[0],a.shape[1]))
    c=b.transpose(1,0)
    return ((b-c)**2).sum(dim=2)