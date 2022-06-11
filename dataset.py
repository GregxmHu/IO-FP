import torch
from tqdm import tqdm
import torch
import random

class Wine():
    def __init__(self,datapath,mode) -> None:
        super().__init__()
        skip=True
        samples=[[], [], []]
        with open(datapath,'r') as f:
            for item in f:
                if skip:
                    skip=False
                    continue
                splits=item.strip('\n').split(',')
                class_id=int(splits[0])
                splits=[float(i) for i in splits[1:]]
                samples[class_id-1].append(splits)
        self.samples=[]
        self.labels=[]
        for i in range(3):
            random.shuffle(samples[i])
            l=int(0.4*len(samples[i]))
            if mode=="train":
                self.labels.extend([i]*len(samples[i][:l]))
                self.samples.extend(samples[i][:l])
            else:
                self.labels.extend([i]*len(samples[i][l:]))
                self.samples.extend(samples[i][l:])
        
    def get(self):
        return torch.tensor(self.samples),self.labels