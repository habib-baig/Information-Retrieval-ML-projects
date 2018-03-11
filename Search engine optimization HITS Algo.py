import json
import numpy as np
from scipy.sparse import csr_matrix
import math

with open('HITS.json') as f:
    tweets = [json.loads(line) for line in f]

userdict={} #to store user_id->index relation
iddict={} #to store the index and user_id relation
cnt=0
for i in range(len(tweets)):
    #rtid=[]
    uid=tweets[i]['user']['id']
    rtid=tweets[i]['retweeted_status']['user']['id']
    if not userdict.has_key(uid):
        userdict[uid]=cnt
        cnt+=1
    if not userdict.has_key(rtid):
        userdict[rtid]=cnt
        cnt+=1
N=cnt
for x in userdict:
    iddict[userdict[x]]=x

graph=np.zeros((N, N))

for i in range(len(tweets)):
    #rtid=[]
    uid=tweets[i]['user']['id']
    rtid=tweets[i]['retweeted_status']['user']['id']
    graph[userdict[uid]][userdict[rtid]]=1 #I am submitting with not weighted graph to have weight jsu this line will change

print graph

hub=np.zeros(N)
auth=np.zeros(N)
hub.fill(1/math.sqrt(N))
auth.fill(1/math.sqrt(N))
#hub.fill(1)
#auth.fill(1)
A=csr_matrix(graph)
At=A.transpose()
for i in range(500):
    hub=A.dot(auth)
    auth=At.dot(hub)
    sum=0
    for x in hub:
        sum+=x*x
    hub[:] = [x / math.sqrt(sum) for x in hub]
    sum=0
    for x in auth:
        sum+=x*x
    auth[:] = [x / math.sqrt(sum) for x in auth]

hub_result={}
for i in range(len(hub)):
    hub_result[iddict[i]]=hub[i]

auth_result={}
for i in range(len(auth)):
    auth_result[iddict[i]]=auth[i]

for w in sorted(hub_result, key=hub_result.get, reverse=True)[:10]:
    print hub_result[w]

for w in sorted(auth_result, key=auth_result.get, reverse=True)[:10]:
    print auth_result[w]
