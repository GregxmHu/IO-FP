import random 
from knn import KnnForWineClassify
import math
from tqdm import tqdm
class TsFeatureSelector():
    def __init__(self,dataset,id2label,k,table_size):
        self.k=k
        self.table_size=table_size
        self.state=13*[0]+13*[1]
        random.shuffle(self.state)
        selected_dim=[j for j in range(13) if self.state[j]==1]
        self.knn=KnnForWineClassify(dataset=dataset,selected_dim=selected_dim,id2label=id2label,k=self.k)
        self.optimal=self.knn.fit()['cls_acc']
        self.optimal_state=self.state.copy()
        self.state_score_trace=[]
        self.state_trace=[]
        m=[list(range(j+1,26)) for j in range(13)]
        self.movement=[]
        self.table=[]
        for i in range(13):
            for j in m[i]:
                self.movement.append((i,j))

    def search(self):
        values=[]
        for move in tqdm(self.movement,desc="searching..."):
            if self.state[move[0]]==self.state[move[1]]:
                continue
            move_state=self.state.copy()
            move_state[move[0]]=self.state[move[1]]
            move_state[move[1]]=self.state[move[0]]
            selected_dim=[j for j in range(13) if move_state[j]==1]
            self.knn.update(selected_dim)
            values.append(
            (move,move_state,self.knn.fit()['cls_acc'])
            )
        values=sorted(values,key=lambda x:x[2],reverse=True)
        for best in values:
            if best[0] in self.table:
                if best[2]>self.optimal:
                    #self.optimal=best[2]
                    self.state=best[1].copy()
                    return best[0]
                else:
                    continue
            else:
                self.state=best[1].copy()
                return best[0]
        
    def run(self):
        for _ in range(100):
            best_move=self.search()
            selected_dim=[j for j in range(13) if self.state[j]==1]
            self.knn.update(selected_dim)
            epoch_optimal=self.knn.fit()['cls_acc']
            if epoch_optimal>self.optimal:
                self.optimal=epoch_optimal
                self.optimal_state=self.state.copy()
                self.table.insert(0,best_move)
                if len(self.table)>self.table_size:
                    self.table=self.table[:self.table_size]
            print("optimal:",self.optimal,"epoch optimal:",epoch_optimal)
            self.state_score_trace.append(epoch_optimal)
            