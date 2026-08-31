"""Iteration 1 data probe: the split/user/item statistics quoted in
experiments/RESULTS.md. Read-only, uses the official data.load().
"""
import sys, time, collections
sys.path.insert(0, '.')
import numpy as np
from data import load, SPLITS
t0=time.time()
sp = load('./KuaiRand-Pure/data')
print('load_s', round(time.time()-t0,1))
for n,v in sp.items():
    y=[x[6] for x in v]
    us=set(x[1] for x in v); vs=set(x[2] for x in v)
    print(n, 'rows',len(v),'pos_rate',round(sum(y)/len(y),4),'users',len(us),'videos',len(vs))
trv=set(x[2] for x in sp['train']); tru=set(x[1] for x in sp['train'])
tra=set(x[3] for x in sp['train'])
for n in ('valid','test'):
    rws=sp[n]
    print(n,'rows w/ unseen video', round(sum(1 for x in rws if x[2] not in trv)/len(rws),4),
          '| unseen user', round(sum(1 for x in rws if x[1] not in tru)/len(rws),4),
          '| unseen author', round(sum(1 for x in rws if x[3] not in tra)/len(rws),4))
# impressions per user distribution in test
c=collections.Counter(x[1] for x in sp['test'])
a=np.array(sorted(c.values()))
print('test impressions/user: median',int(np.median(a)),'mean',round(a.mean(),1),'p90',int(np.percentile(a,90)),'max',a.max())
# per-user pos count
pu=collections.Counter(); iu=collections.Counter()
for x in sp['test']:
    iu[x[1]]+=1; pu[x[1]]+=x[6]
zero=sum(1 for u in iu if pu[u]==0); allpos=sum(1 for u in iu if pu[u]==iu[u])
print('test users', len(iu), 'zero-pos', zero, 'all-pos', allpos)
# tab distribution
print('tab counts train', collections.Counter(x[4] for x in sp['train']).most_common(8))