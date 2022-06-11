from click import option
import torch
import torch.nn as nn
import math
import random

class KnnForWineClassify():
    def __init__(self, dataset, selected_dim,id2label,k):
        self.k=k
        self.origin_dataset=dataset
        self.dataset=self.origin_dataset[:,selected_dim]   #selected_dim
        self.dp_mat=self.dataset@self.dataset.transpose(0,1)    # dot-score for sample-pair
        self.id2label=id2label      #map id to label
        self.state_score_trace=[]
        # self.dp_mat: nxn
        #self.id2label id2label[id]=class_of(id)
    def fit(self):

        topk_id=torch.topk(self.dp_mat,k=self.k,dim=1,largest=True).indices
        topk_id=[  [id.item() for id in topk_id[row_id] if id.item() != row_id]
                        for row_id in topk_id.shape[0]
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
        self.dp_mat=self.dataset@self.dataset.transpose(0,1)    # dot-score for sample-pair

class TsFeatureSelector():
    def __init__(self):
        pass

class SaFeatureSelector():
    def __init__(self,dataset,id2label,k,T_k,gamma):
        self.T_k=T_k
        self.k=k
        self.gamma=gamma
        self.state=[3,5,7,9]
        self.knn=KnnForWineClassify(dataset=dataset,selected_dim=self.state,id2label=id2label,k=self.k)
        self.total_num_dims=13
        self.state_score=self.knn.fit()
        self.optimal=self.state_score
        self.dim_set=list(range(self.total_num_dims))

    def search(self):
        # run a period to down temperature
        new_num_dim=random.choice(list(range(self.total_num_dims)))+1
        new_option=random.sample(self.dim_set,new_num_dim)
        old_option=random.sample(self.dim_set,len(self.state))
        option=random.choice([old_option,new_option])
        return option

    def run(self):
        while self.T>0.01:
            for _ in range(50):
                self.state_score_trace.append(self.state_score)
                new_state=self.search()
                new_state_score=self.knn.fit()['cls_acc']
                if new_state_score>self.state_score:
                    self.state=new_state
                    self.state_score=new_state_score
                    self.optimal=new_state
                else:
                    kesi=random.uniform(0,1)
                    delta_f=self.state_score-new_state_score
                    prob=math.exp(-delta_f/self.T_k)
                    if prob>kesi:
                        self.state_score=new_state_score
                        self.state=new_state
            self.T*=self.gamma

