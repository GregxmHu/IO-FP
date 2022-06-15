import select
import sys
from TS import TsFeatureSelector
from knn import KnnForWineClassify
from util import set_seed
from dataset import Wine
from SA import SaFeatureSelector
from matplotlib import pyplot as plt
stor=sys.argv[1]
ds=Wine("wine_3cls.csv","train")
dataset,id2label=ds.get()
# train sa
if stor=="sa":
    sa_selector=SaFeatureSelector(
        dataset=dataset,
        id2label=id2label,
        k=4,
        T_k=1000,
        gamma=0.95
        )
    sa_selector.run()
    sa_dim=[j for j in range(13) if sa_selector.optimal_state[j]==1]
    with open("sa_optimal.txt",'a') as f:
        f.write(str(sa_selector.optimal)+"\t"+str(sa_dim)+"\n")
    plt.figure()
    plt.plot(sa_selector.state_score_trace)
    plt.xlabel("step")
    plt.ylabel("acc")
    plt.savefig("SA_score.jpg")


else:
    ts_selector=TsFeatureSelector(
        dataset=dataset,
        id2label=id2label,
        k=4,
        table_size=10
    )

    ts_selector.run()
    ts_dim=[j for j in range(13) if ts_selector.optimal_state[j]==1]
    with open("ts_optimal.txt",'a') as f:
        f.write(str(ts_selector.optimal)+"\t"+str(ts_dim)+"\n")
    plt.figure()
    plt.plot(ts_selector.state_score_trace)
    plt.xlabel("step")
    plt.ylabel("acc")
    plt.savefig("TS_score.jpg")

#if stor=="ts":
#    ts_dim=[j for j in range(13) if ts_selector.optimal_state[j]==1]
#else:
#    sa_dim=[j for j in range(13) if sa_selector.optimal_state[j]==1]

### inference
#ds=Wine("wine_3cls.csv","test")
#dataset,id2label=ds.get()
#if stor=="ts":
#    ts_knn=KnnForWineClassify(dataset=dataset,selected_dim=ts_dim,id2label=id2label,k=4)
#    print(ts_dim)
#    print(ts_knn.fit())
#else:
#    sa_knn=KnnForWineClassify(dataset=dataset,selected_dim=sa_dim,id2label=id2label,k=4)
#    print(sa_dim)
#    print(sa_knn.fit())
