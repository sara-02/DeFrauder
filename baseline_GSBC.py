import networkx as nx
import json
import sys
import time
import ast
import csv
import random
import numpy as np
from datetime import datetime
import math
from sklearn.cluster import KMeans
import operator
import pandas as pd
import re
from collections import Counter
from sklearn.linear_model.logistic import LogisticRegression
# from CCgroups import reviewtightness,producttightness,productreviewerratio,neighbortightness,averagetimewindow_ratingvariance,groupsize,multiplereview

# G = nx.path_graph(4)
# G.add_path([10, 11, 12])
# for c in sorted(nx.connected_components(G), key=len, reverse=True):

# sys.exit(0)


class Groups:
    def __init__(self, users, prods):
        self.users = users
        self.prods = prods

    def __lt__(self, other):
        return len(self.users) < len(other.users)


class Review:
    def __init__(self, userid, useridmapped, prodid, prodidmapped, rating,
                 label, date, content):
        self.userid = userid
        self.useridmapped = useridmapped
        self.prodid = prodid
        self.prodidmapped = prodidmapped
        self.rating = rating
        self.label = label
        self.date = date
        self.content = content

    def __repr__(self):
        return '({})'.format(self.prodid)

    def __hash__(self):
        return hash(self.prodid)

    def __eq__(self, other):
        return self.prodid == other.prodid

    def __ne__(self, other):
        return not self.__eq__(other)


userIdMapping = {}
userIdMappingfile = open("userIdMapping", 'r')
for row in userIdMappingfile:

    userid = row.split("\t")[1].strip()
    useridmapped = row.split("\t")[0].strip()

    if userid not in userIdMapping:
        userIdMapping[userid] = 0
    userIdMapping[userid] = useridmapped

productIdMapping = {}
productIdMappingfile = open("productIdMapping", 'r')
for row in productIdMappingfile:

    prodid = row.split("\t")[1].strip()
    prodidmapped = row.split("\t")[0].strip()
    if prodid not in productIdMapping:
        productIdMapping[prodid] = 0
    productIdMapping[prodid] = prodidmapped

allprods = {}
allusers = {}
reviewtime = {}
reviewrating = {}
reviewcontent = {}
minn = {}
d = {}
fake = set()
filee = open("metadata", 'r')
for f in filee:

    fsplit = f.split("\t")

    userid = int(fsplit[0])
    try:
        useridmapped = userIdMapping[fsplit[0]]
    except:
        useridmapped = ''

    prodid = int(fsplit[1])
    try:
        prodidmapped = productIdMapping[fsplit[1]]
    except:
        prodidmapped = ''

    rating = int(round(float(fsplit[2])))

    label = fsplit[3]
    if int(label) == -1:
        fake.add(userid)
    date = fsplit[4].strip()
    if prodid not in d:
        minn[prodid] = 0
        d[prodid] = datetime.strptime(date, "%Y-%m-%d").date()

    minn[prodid] = datetime.strptime(date, "%Y-%m-%d").date()
    if minn[prodid] < d[prodid]:
        d[prodid] = minn[prodid]

filee.close()

text = []
filee = open("reviewContent", 'r')
for f in filee:
    fsplit = f.split("\t")
    text.append(fsplit[3].strip())

filee.close()

wholerev = {}
c = 0
filee = open("metadata", 'r')
l = []
for f in filee:

    fsplit = f.split("\t")

    userid = int(fsplit[0])
    try:
        useridmapped = userIdMapping[fsplit[0]]
    except:
        useridmapped = ''

    prodid = int(fsplit[1])
    try:
        prodidmapped = productIdMapping[fsplit[1]]
    except:
        prodidmapped = ''

    rating = int(round(float(fsplit[2])))
    label = fsplit[3]
    date = fsplit[4].strip()

    newdate = datetime.strptime(date, "%Y-%m-%d").date()

    datetodays = (newdate - d[prodid]).days

    r = Review(userid, useridmapped, prodid, prodidmapped, rating, label,
               datetodays, text[c])

    if userid not in reviewtime:
        reviewtime[userid] = {}
    if prodid not in reviewtime[userid]:
        reviewtime[userid][prodid] = datetodays
    if userid not in reviewrating:
        reviewrating[userid] = {}
    if prodid not in reviewrating[userid]:
        reviewrating[userid][prodid] = rating
    if userid not in reviewcontent:
        reviewcontent[userid] = {}
    if prodid not in reviewcontent[userid]:
        reviewcontent[userid][prodid] = text[c]
    if userid not in allusers:
        allusers[userid] = []
    if prodid not in allprods:
        allprods[prodid] = []
    if userid not in wholerev:
        wholerev[userid] = {}
    if prodid not in wholerev[userid]:
        wholerev[userid][prodid] = r

    allprods[prodid].append(r)
    # l.append(len(allprods[prodid]))
    allusers[userid].append(r)

    c = c + 1

filee.close()
# l.sort(reverse=True)
# print l[0]
# sys.exit(0)


def reviewtightness(group, L):
    v = 0
    for user in group.users:
        for prod in group.prods:
            # prod=prod.split("_")[0]
            if prod in reviewtime[user]:
                v = v + 1
    return (v * L) / (1.0 * len(group.users) * len(group.prods))

# def neighbortightness(group,L):

#      userlist=list(group.users)
#      denom=0
#    num=0
#      for user1i in range(len(userlist)):
#         user1=userlist[user1i]
#         for user2i in range(user1i+1,len(userlist)):
#             user2=userlist[user2i]

#           union=0
#           intersection=0
#             for prod in group.prods:
#               if prod in reviewtime[user1] and prod in reviewtime[user2]:
#                   union=union+1
#               if prod in reviewtime[user1] or prod in reviewtime[user2]:
#                   intersection=intersection+1

#             # union=set(allusers[user1]).union(set(allusers[user2]))
#             # intersection=set(allusers[user1]).intersection(set(allusers[user2]))
#             num=num+len(intersection)/(len(union)*1.0)
#             denom=denom+1

#      return (num*L)/(1.0*denom)

# def producttightness(group):

#      c=0
#      userlist=list(group.users)
#      prodlist={}
#      for user in userlist:
#       if user not in prodlist:
#           prodlist[user]=set()
#       for prod in group.prods:
#           if prod in reviewtime[user]:
#               prodlist[user].add(prod)

#     for user in userlist:
#         if c==0:
#             intersection=set(prodlist[user])
#             # union= set(prodlist[user])
#         else:
#           intersection=intersection.intersection(set(prodlist[user]))
#               # union=union.union(set(prodlist[user]))
#         c=c+1

#      return len(intersection)/(len(group.prods)*1.0)


def neighbortightness(group, L):

    userlist = list(group.users)
    denom = 0
    num = 0
    for user1i in range(len(userlist)):
        user1 = userlist[user1i]
        for user2i in range(user1i + 1, len(userlist)):
            user2 = userlist[user2i]
            union = set(allusers[user1]).union(set(allusers[user2]))
            intersection = set(allusers[user1]).intersection(
                set(allusers[user2]))
            num = num + len(intersection) / (len(union) * 1.0)
            denom = denom + 1

    return (num * L) / (1.0 * denom)


def producttightness(group):

    c = 0
    userlist = list(group.users)
    for user in userlist:
        if c == 0:
            intersection = set(allusers[user])
            union = set(allusers[user])
        else:
            intersection = intersection.intersection(set(allusers[user]))
            union = union.union(set(allusers[user]))
        c = c + 1

    return len(intersection) / (len(union) * 1.0)


def averagetimewindow_ratingvariance(group, L):

    avg = 0
    var = 0
    for prod in group.prods:
        prodlist = []
        prodtym = []
        # prod=prod.split("_")[0]
        minn = float('inf')
        maxx = 0
        for user in group.users:
            if prod in reviewtime[user]:
                prodlist.append(reviewrating[user][prod])
                prodtym.append(reviewtime[user][prod])
                # if reviewtime[user][prod]<minn:
                #     minn=reviewtime[user][prod]
                # if reviewtime[user][prod]>maxx:
                #     maxx=reviewtime[user][prod]

        var = var + np.var(prodlist)
        ans = np.std(prodtym)
        if ans <= 30:
            avg = avg + (1 - ans / 30.0)

    var = var / (-1.0 * len(group.prods))
    rating_variance = 2 * (1 - (1.0 / (1 + math.exp(var))))

    return (avg * L) / (1.0 * len(group.prods)), rating_variance * L
    # return rating_variance*L[group]


def productreviewerratio(group):

    maxx = 0

    for prod in group.prods:

        num = 0
        denom = 0
        for user in group.users:
            if prod in reviewtime[user]:
                num = num + 1

        for r in allprods[prod]:
            # if int(r.rating)==int(prod.split("_")[1]):
            denom = denom + 1

        ans = num / (1.0 * denom)
        if ans > maxx:
            maxx = ans

    return maxx


def multiplereview(group):

    return 0
    ans = 0
    tot = 0
    for p in group.prods:
        # prod=p.split("_")[0]
        prod = p
        c = 0
        for rev in allprods[prod]:
            if rev.userid in group.users:
                c = c + 1
                tot = tot + 1
        if c > 1:
            ans = ans + c
    return (ans * 1.0) / (tot)


def groupsize(group):
    return 1 / (1 + math.exp(3 - len(group.users)))


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)


def cosine(content_a, content_b):

    text1 = content_a
    text2 = content_b

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine_result = get_cosine(vector1, vector2)
    return cosine_result


def GCS(group):

    maxx = 0
    for prod in group.prods:
        avg = 0
        c = 0
        userlist = list(group.users)
        for r1i in range(len(userlist)):
            r1 = userlist[r1i]
            if prod in reviewtime[r1]:
                for r2i in range(r1i + 1, len(userlist)):
                    r2 = userlist[r2i]
                    if prod in reviewtime[r2]:
                        avg = avg + cosine(reviewcontent[r1][prod],
                                           reviewcontent[r2][prod])
                        c = c + 1
        if c == 0:
            avg = 0
        else:
            avg = avg / (c * 1.0)
        if avg > maxx:
            maxx = avg

    return maxx


def GMCS(group):

    avg = 0
    totc = len(group.users)

    if len(group.prods) == 1:
        return 1
    for user in group.users:
        ans = 0
        c = 0
        prodlist = list(group.prods)
        for p1i in range(len(prodlist)):
            p1 = prodlist[p1i]

            if p1 in reviewtime[user]:
                for p2i in range(p1i + 1, len(prodlist)):
                    p2 = prodlist[p2i]
                    if p2 in reviewtime[user]:
                        ans = ans + cosine(reviewcontent[user][p1],
                                           reviewcontent[user][p2])
                        c = c + 1
        if c != 0:
            avg = avg + (ans * 1.0) / c
        else:
            totc = totc - 1

    if totc == 0:
        return 0
    return (avg * 1.0) / (totc)


def ConstructReviewerGraph():
    G = nx.Graph()
    theta = 0.2
    MP = 10000
    tau = 10.0
    delta = 0.4
    suspicion = {}
    adj = {}
    ans = {}
    refinedgroups = []

    for prod in allprods:
        if prod not in suspicion:
            suspicion[prod] = 0
        suspicion[prod] = (2.0 / (1 + math.exp(
            -1 *
            (pow(abs(MP - len(allprods[prod])), theta)) + pow(2, theta)))) - 1

        userlist = set()
        for u in allprods[prod]:
            userlist.add(u.userid)
            G.add_node(u.userid)
        userlist = list(userlist)
        for u1i in range(len(userlist)):
            u1 = userlist[u1i]
            if u1 not in ans:
                ans[u1] = {}
                adj[u1] = 0
            for u2i in range(u1i + 1, len(userlist)):
                u2 = userlist[u2i]
                if u2 not in ans[u1]:
                    ans[u1][u2] = 0
                if abs(reviewtime[u1][prod] - reviewtime[u2][prod]
                       ) <= tau and abs(reviewrating[u1][prod] -
                                        reviewrating[u2][prod]) < 2:
                    ans[u1][u2] = ans[u1][u2] + 0.5 * suspicion[prod] * (
                        (1 -
                         (abs(reviewtime[u1][prod] - reviewtime[u2][prod])) /
                         tau) + (1 - (abs(reviewrating[u1][prod] -
                                          reviewrating[u2][prod])) / 2.0))

    for u1 in ans:
        if u1 not in adj:
            adj[u1] = 1
        for u2 in ans[u1]:
            if u2 in adj and adj[u2] == 0:
                jaccard = (1.0 * len(
                    set(allusers[u1]).intersection(set(allusers[u2])))) / (len(
                        set(allusers[u1]).union(set(allusers[u2]))))
                ans[u1][u2] = ans[u1][u2] * jaccard
                similarity = (2.0 / (1 + math.exp(-1 * ans[u1][u2]))) - 1
                if similarity >= delta:
                    G.add_edge(u1, u2)

    return G


def createGraph(comp, G):
    Gsub = nx.Graph()
    for n1 in comp:
        Gsub.add_node(n1)
        for n2 in comp:
            if n1 != n2:
                Gsub.add_node(n2)
                if G.has_edge(n1, n2):
                    Gsub.add_edge(n1, n2)
    return Gsub


def calc_score(g, Lsub):
    score = []
    ans = averagetimewindow_ratingvariance(g, Lsub)
    score = [
        reviewtightness(g, Lsub),
        neighbortightness(g, Lsub),
        producttightness(g),
        ans[0],
        ans[1],
        productreviewerratio(g),
        groupsize(g),
        multiplereview(g)
    ]
    score2 = score
    score.append(GCS(g))
    score.append(GMCS(g))
    return score, sum(score2) / (len(score2) * 1.0)


def SS(bcc, prods):
    c = 0
    prods = set()
    for u1 in bcc:
        c = c + 1
        if c == 1:
            for p in allusers[u1]:
                prods.add(p.prodid)
        else:
            for p in allusers[u1]:
                prods.add(p.prodid)

    group = Groups(bcc, prods)
    Lsub = 1.0 / (1 + (math.exp(3 - len(group.users) - len(group.prods))))
    ans = calc_score(group, Lsub)
    return ans[0], ans[1], group


def FindGroups(Gsub):
    MAXSIZE = 50
    MINSPAM = 0.58  #NYC
    # MINSPAM=0.59 #ZIP

    for bcc in list(nx.biconnected_components(Gsub)):
        c = 0
        denom = 0
        bcc = map(int, bcc)
        if len(bcc) > MAXSIZE:
            bccgraph = createGraph(bcc, Gsub)
            if nx.is_connected(bccgraph):
                mincutedge = nx.minimum_edge_cut(bccgraph)
                for e in mincutedge:
                    bccgraph.remove_edge(e[0], e[1])
                cc = nx.connected_components(bccgraph)
                for cx in cc:
                    FindGroups(createGraph(cx, bccgraph))
        else:
            ans = SS(bcc)
            scorepred = ans[0]
            spamicity = ans[1]
            group = ans[2]

            if spamicity >= MINSPAM:
                if str(bcc) not in ccgroups:
                    ccgroups[str(bcc)] = 0

                c = 0
                denom = 0
                for u in group.users:
                    if u in fake:
                        c = c + 1
                    denom = denom + 1
                # ccgroups[str(finalgrps[grp]['users'])]=spamicity
                store = (c * 1.0) / denom

                c = 0
                denom = 0
                for u in group.users:
                    for p in group.prods:
                        if p in wholerev[u]:
                            if int(wholerev[u][p].label) == -1:
                                c = c + 1
                            denom = denom + 1
                ccgroups[str(bcc)] = spamicity
                if len(ccgroups) not in grps:
                    grps[len(ccgroups)] = {
                        'id': len(ccgroups),
                        'users': list(group.users),
                        'prods': list(group.prods),
                        'scorepred': scorepred,
                        'scoregt': store,
                        'scoregtreviewprec': (c * 1.0) / denom,
                        'fakegt': 0,
                        'fakepred': spamicity
                    }

                # filew.write(str(bcc)+"\n")
            elif len(bcc) > 2:
                bccgraph = createGraph(bcc, Gsub)
                if nx.is_connected(bccgraph):
                    mincutedge = nx.minimum_edge_cut(bccgraph)
                    for e in mincutedge:
                        bccgraph.remove_edge(e[0], e[1])
                    cc = nx.connected_components(bccgraph)
                    for cx in cc:
                        FindGroups(createGraph(cx, bccgraph))


def weighted_params():
    with open('../../../../baselines/GSBC/YELPNYC/finalgrps.json', 'r') as fp:
        grps = json.load(fp)
        X = []
        Y = []
        c = 0
        tot = 0
        mc = 0
        # for grp in grps:
        #     scorepred=grps[grp]['scorepred']
        #     if len(grps[grp]['users'])>1 and (sum(scorepred[:5]))/6.0>0.4:
        #         tot=tot+1
        for grp in grps:
            scorepred = grps[grp]['scorepred']
            if len(grps[grp]['users']) > 1:
                mc = mc + 1
        # mc=len(grps)
        for grp in grps:
            scorepred = grps[grp]['scorepred']
            if len(grps[grp]['users']) > 1:
                X.append(grps[grp]['scorepred'][:-2])
                if c < int(0.72 * (mc)):
                    Y.append(0)
                else:
                    Y.append(1)
                c = c + 1
        classifier = LogisticRegression()
        classifier.fit(X, Y)

        for grp in grps:
            scorepred = grps[grp]['scorepred']
            if len(grps[grp]['users']) > 1:
                if grps[grp]['id'] not in ccgroups:
                    ccgroups[grps[grp]['id']] = 0
                    gtgroups[grps[grp]['id']] = 0
                ccgroups[grps[grp]['id']] = sum([
                    grps[grp]['scorepred'][i] * classifier.coef_[0][i]
                    for i in range(8)
                ]) / 8.0
                gtgroups[grps[grp]['id']] = grps[grp]['scoregt']


# bcc=set(['127479', '126974'])
# bcc=map(int, bcc)
# SS(bcc)
# sys.exit(0)

ccgroups = {}
gtgroups = {}
weighted_params()
# grps2={}
# G=ConstructReviewerGraph()
# for c in nx.connected_components(G):
#     if len(c)>1:
#         Gsub=createGraph(c,G)
#         FindGroups(Gsub)


def dcg(r_baseline, r_gt):
    ansb = 0
    ansg = 0
    c = 0
    for g in range(len(r_baseline)):
        ansb = ansb + (r_baseline[g][1] * 1.0) / math.log(
            r_baseline[g][0] + 1, 2)
        ansg = ansg + (r_gt[g][1] * 1.0) / math.log(r_gt[g][0] + 1, 2)
        c = c + 1
        if c % 10 == 0:
            print ansb / (ansg * 1.0)


r_baseline = []
precs = []
num = 0
denom = 0
score = {}
gt_id = []
cc_id = []
sorted_ccgroups = sorted(ccgroups.items(),
                         key=operator.itemgetter(1),
                         reverse=True)
# filew=open('baseline_GSBC_algo.txt','w')
for cc in sorted_ccgroups:
    # filew.write(str(cc[0])+"\t"+str(cc[1])+"\t"+str(gtgroups[cc[0]])+"\n")
    if cc[0] not in score:
        score[cc[0]] = 0
    score[cc[0]] = gtgroups[cc[0]]
    num = num + gtgroups[cc[0]]
    denom = denom + 1
    # prec=num/(denom*1.0)
    # if denom%10==0:
    #     precs.append(num/(denom*1.0))
    r_baseline.append((denom, gtgroups[cc[0]]))
    cc_id.append(cc[0])
    if denom == 100:
        break
# filew.close()

r_gt = []
sorted_score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)
c = 0
for s in sorted_score:
    c = c + 1
    r_gt.append((c, s[1]))
    gt_id.append(s[0])

inter = 0
for i in range(0, 100, 10):
    inter = len(set(cc_id[:i + 10]).intersection(set(gt_id[:i + 10])))
    print inter / ((i + 10) * 1.0)
print '##################'
dcg(r_baseline, r_gt)

# dcgg=dcg(r_baseline)
# idcg=dcg(r_gt)
# print str(round(dcgg,2))+'/'+str(round(idcg,2))+'='+str(round(dcgg/(1.0*idcg),2))

# with open('baseline_fim.json', 'r') as fp:
#     grps = json.load(fp)
#     x=0
#     for grp in grps:
#         bcc=map(int,grps[grp]['users'])
#         ans=SS(bcc,map(int,grps[grp]['prods']))
#         scorepred=ans[0]
#         spamicity=ans[1]
#         group=ans[2]

#         # if str(bcc) not in ccgroups:
#         #     ccgroups[str(bcc)]=0

#         c=0
#         denom=0
#         for u in group.users:
#             if u in fake:
#                 c=c+1
#             denom=denom+1
#         # ccgroups[str(finalgrps[grp]['users'])]=spamicity
#         store=(c*1.0)/denom

#         c=0
#         denom=0
#         for u in group.users:
#             for p in group.prods:
#                 if p in wholerev[u]:
#                     if int(wholerev[u][p].label)==-1:
#                         c=c+1
#                     denom=denom+1
#         # ccgroups[str(bcc)]=spamicity
#         if grps[grp]['id'] not in grps2:
#             grps2[grps[grp]['id']]={'id':grps[grp]['id'],'users':list(group.users),'prods':list(group.prods),'scorepred':scorepred, 'scoregt':store, 'scoregtreviewprec':(c*1.0)/denom, 'fakegt':0,'fakepred':spamicity}

#         x=x+1
#         if x==10000:
#             break

# print precs
# with open('baseline_fim2.json', 'w') as fp:
#     json.dump(grps2, fp)
print 'end'