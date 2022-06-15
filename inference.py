import select
import sys
from TS import TsFeatureSelector
from knn import KnnForWineClassify
from util import set_seed
from tqdm import tqdm
from dataset import Wine
from SA import SaFeatureSelector
from matplotlib import pyplot as plt
ds=Wine("wine_3cls.csv","test")
dataset,id2label=ds.get()
dim=[6, 0, 10, 11, 2, 5]
rel=[]
for k in tqdm(range(2,100)):
    knn=KnnForWineClassify(dataset=dataset,selected_dim=dim,id2label=id2label,k=k)
    results=knn.fit()
    rel.append((results['cls_acc'],k))
    #print(dim)
    #print(knn.fit())
with open("sa_k_s.txt",'w') as f:
    for item in rel:
        f.write(str(item[0])+"\t"+str(item[1])+"\n")