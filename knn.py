from click import option
import torch
import torch.nn as nn
import math
import random
from tqdm import tqdm
from util import cal_distance


class KnnForWineClassify():
    def __init__(self, dataset, selected_dim,id2label,k):
        self.k=k
        self.origin_dataset=dataset
        self.dataset=self.origin_dataset[:,selected_dim]  #selected_dim
        self.dp_mat=cal_distance(self.dataset)    # dot-score for sample-pair
        self.id2label=id2label      #map id to label

    def fit(self):

        topk_id=torch.topk(self.dp_mat,k=self.k,dim=1,largest=False).indices
        topk_id=[  [id.item() for id in topk_id[row_id] if id.item() != row_id]
                        for row_id in range(topk_id.shape[0])
        ]
        topk_label=[  [self.id2label[id] for id in row] 
                        for row in topk_id
                    ]
        results=[max(set(row),key=row.count) for row in topk_label]
        acc=0
        for j in range(len(results)):
            if self.id2label[j]==results[j]:
                acc+=1
        return  {
            "topk_label":topk_label,
            "cls_results":results,
            "cls_acc":acc/len(results)
        }
        
        
    def update(self,selected_dim):

        self.dataset=self.origin_dataset[:,selected_dim]   #selected_dim
        self.dp_mat=self.dp_mat=cal_distance(self.dataset)    # dot-score for sample-pair


