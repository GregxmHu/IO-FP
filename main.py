import select
import sys
from TS import TsFeatureSelector
from knn import KnnForWineClassify
from util import set_seed
from dataset import Wine
from SA import SaFeatureSelector
from matplotlib import pyplot as plt
ds=Wine("wine_3cls.csv","train")
dataset,id2label=ds.get()
# train sa
sa_selector=SaFeatureSelector(
    dataset=dataset,
    id2label=id2label,
    k=4,
    T_k=1000,
    gamma=0.95
    )
sa_selector.run()
plt.figure()
plt.plot(sa_selector.state_score_trace)
plt.xlabel("step")
plt.ylabel("acc")
plt.savefig("SA_score.jpg")



ts_selector=TsFeatureSelector(
    dataset=dataset,
    id2label=id2label,
    k=4,
    table_size=10
)

ts_selector.run()

plt.figure()
plt.plot(ts_selector.state_score_trace)
plt.xlabel("step")
plt.ylabel("acc")
plt.savefig("TS_score.jpg")


ts_dim=[j for j in range(13) if ts_selector.optimal_state[j]==1]
sa_dim=[j for j in range(13) if sa_selector.optimal_state[j]==1]

### inference
ds=Wine("wine_3cls.csv","test")
dataset,id2label=ds.get()
ts_knn=KnnForWineClassify(dataset=dataset,selected_dim=ts_dim,id2label=id2label,k=4)
sa_knn=KnnForWineClassify(dataset=dataset,selected_dim=ts_dim,id2label=id2label,k=4)

print(ts_knn.fit())
print(sa_knn.fit())