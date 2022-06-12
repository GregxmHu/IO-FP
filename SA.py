import random 
from knn import KnnForWineClassify
import math
class SaFeatureSelector():
    def __init__(self,dataset,id2label,k,T_k,gamma):
        self.T_k=T_k
        self.k=k
        self.gamma=gamma
        self.state=13*[0]+13*[1]
        random.shuffle(self.state)
        selected_dim=[j for j in range(13) if self.state[j]==1]
        self.knn=KnnForWineClassify(dataset=dataset,selected_dim=selected_dim,id2label=id2label,k=self.k)
        self.state_score=self.knn.fit()['cls_acc']
        self.optimal=self.state_score
        self.optimal_state=self.state.copy()
        self.state_score_trace=[]
        self.state_trace=[]
    def search(self):
        while True:
            i,j=random.choices(range(26),k=2)
            if i>=13 and j>=13:
                continue
            if self.state[i]==self.state[j]:
                continue
            if 1 not in self.state[:13]:
                continue
            break
        new_state=self.state.copy()
        new_state[i]=self.state[j]
        new_state[j]=self.state[i]
        return new_state



    def run(self):
        while self.T_k>0.001:
            sub_optimal=self.state_score
            sub_optimal_state=self.state.copy()

            for i in range(100):
                self.state_score_trace.append(self.state_score)
                self.state_trace.append(self.state[:13])
                new_state=self.search()
                selected_dim=[j for j in range(13) if new_state[j]==1]
                self.knn.update(selected_dim)
                new_state_score=self.knn.fit()['cls_acc']

                if new_state_score>self.state_score:

                    self.state=new_state.copy()
                    self.state_score=new_state_score

                    if self.state_score>sub_optimal:
                        sub_optimal=new_state_score
                        sub_optimal_state=new_state.copy()

                else:
                    kesi=random.uniform(0,1)
                    delta_f=self.state_score-new_state_score
                    prob=math.exp(-delta_f/self.T_k)
                    if prob>kesi:
                        self.state_score=new_state_score
                        self.state=new_state.copy()
                    else:
                        selected_dim=[j for j in range(13) if self.state[j]==1]
                        self.knn.update(selected_dim)
            self.T_k*=self.gamma
            if sub_optimal>self.optimal:
                self.optimal=sub_optimal
                self.optimal_state=sub_optimal_state.copy()
            print("current_dim",[j for j in range(13) if self.optimal_state[j]==1],"current_optimal",self.optimal)