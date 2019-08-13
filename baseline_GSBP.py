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


G = nx.Graph()
allprods = {}
allusers = {}
reviewtime = {}
reviewrating = {}
reviewcontent = {}
rnoofrevs = {}
pnoofrevrs = {}
minn = {}
d = {}
c = 0
fake = set()
filee = open("metadata", 'r')
for f in filee:

    fsplit = f.split("\t")

    userid = int(fsplit[0])

    prodid = int(fsplit[1])

    rating = int(round(float(fsplit[2])))

    label = fsplit[3]
    if int(label) == -1:
        fake.add(userid)
    date = fsplit[4].strip()

    if userid not in rnoofrevs:
        rnoofrevs[userid] = 0
    rnoofrevs[userid] = rnoofrevs[userid] + 1
    if prodid not in d:
        pnoofrevrs[prodid] = set()
        minn[prodid] = 0
        d[prodid] = datetime.strptime(date, "%Y-%m-%d").date()

    minn[prodid] = datetime.strptime(date, "%Y-%m-%d").date()
    if minn[prodid] < d[prodid]:
        d[prodid] = minn[prodid]
    pnoofrevrs[prodid].add(userid)
    # if c==1000:
    #   break

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
    prodid = int(fsplit[1])

    if rnoofrevs[userid] > 2 and len(pnoofrevrs[prodid]) > 1:
        rating = int(round(float(fsplit[2])))
        label = fsplit[3]
        date = fsplit[4].strip()

        newdate = datetime.strptime(date, "%Y-%m-%d").date()

        datetodays = (newdate - d[prodid]).days

        r = Review(userid, '', prodid, '', rating, label, datetodays, text[c])

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

        if userid not in wholerev:
            wholerev[userid] = {}
        if prodid not in wholerev[userid]:
            wholerev[userid][prodid] = r

        G.add_node(userid)
        if userid not in allusers:
            allusers[userid] = set()
        if prodid not in allprods:
            allprods[prodid] = set()

        allprods[prodid].add(userid)
        allusers[userid].add(prodid)
        c = c + 1
        # if c==1000:
        #   break

filee.close()

del rnoofrevs
del pnoofrevrs

for p in allprods:
    allprods[p] = list(allprods[p])
for u in allusers:
    allusers[u] = list(allusers[u])

edgeweight = {}
tau = 30
MAXSIZE = 30
delta = 0.4
for p in allprods:
    for r1i in range(len(allprods[p])):
        r1 = allprods[p][r1i]
        if r1 not in edgeweight:
            edgeweight[r1] = {}
        for r2i in range(r1i + 1, len(allprods[p])):
            r2 = allprods[p][r2i]
            if r2 not in edgeweight[r1]:
                edgeweight[r1][r2] = 0
            if abs(reviewtime[r1][p] - reviewtime[r2][p]) <= tau:
                edgeweight[r1][r2] = edgeweight[r1][r2] + 1

for n1i in range(len(G.nodes())):
    n1 = list(G.nodes())[n1i]
    for n2i in range(n1i + 1, len(G.nodes())):
        n2 = list(G.nodes())[n2i]
        if n1 in edgeweight and n2 in edgeweight[n1]:
            G.add_edge(n1, n2, weight=edgeweight[n1][n2])
        elif n2 in edgeweight and n1 in edgeweight[n2]:
            G.add_edge(n1, n2, weight=edgeweight[n2][n1])


def reviewtightness(group, L):
    v = 0
    for user in group.users:
        for prod in group.prods:
            if prod in reviewtime[user]:
                v = v + 1
    return (v * L) / (1.0 * len(group.users) * len(group.prods))


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

    avgtw = 0
    avger = 0
    var = 0
    for prod in group.prods:
        prodlist = []
        prodtym = []
        minn = float('inf')
        maxx = 0
        for user in group.users:
            if prod in reviewtime[user]:
                prodlist.append(reviewrating[user][prod])
                prodtym.append(reviewtime[user][prod])
                if reviewtime[user][prod] < minn:
                    minn = reviewtime[user][prod]
                if reviewtime[user][prod] > maxx:
                    maxx = reviewtime[user][prod]

        var = var + np.var(prodlist)
        ans = maxx - minn
        if ans <= 30:
            avgtw = avgtw + (1 - ans / 30.0)
        if ans <= 180:
            avger = avger + (1 - ans / 180.0)

    var = var / (-1.0 * len(group.prods))
    rating_variance = 2 * (1 - (1.0 / (1 + math.exp(var))))

    return (avgtw * L) / (1.0 * len(group.prods)), (avger * L) / (
        1.0 * len(group.prods)), rating_variance * L


def productreviewerratio(group):

    maxx = 0

    for prod in group.prods:

        num = 0
        denom = 0
        for user in group.users:
            if prod in reviewtime[user]:
                num = num + 1

        for r in allprods[prod]:
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


def SS(bcc):
    c = 0
    prods = set()
    for u1 in bcc:
        prods = prods.union(allusers[u1])
    group = Groups(bcc, prods)
    Lsub = 1.0 / (1 + (math.exp(3 - len(group.users) - len(group.prods))))
    ans = calc_score(group, Lsub)
    return ans[0], ans[1], group


def calc_score(g, Lsub):
    score = []
    ans = averagetimewindow_ratingvariance(g, Lsub)
    score = [
        reviewtightness(g, Lsub),
        neighbortightness(g, Lsub),
        producttightness(g), ans[0], ans[1],
        productreviewerratio(g), ans[2],
        groupsize(g)
    ]
    score2 = score
    score.append(GCS(g))
    score.append(GMCS(g))
    return score, sum(score2) / (len(score2) * 1.0)


def drop(G, k=1):
    G2 = G.copy()

    for edge in G.edges():
        if G2[edge[0]][edge[1]]['weight'] < k:
            G2.remove_edge(edge[0], edge[1])
    return G2


def FindGroups(G, k):
    ansmain = 0
    scorepredmain = 0
    spamicitymain = 0
    groupmain = 0

    if len(G.nodes()) <= MAXSIZE:
        ansmain = SS(list(G.nodes()))
        scorepredmain = ansmain[0]
        spamicitymain = ansmain[1]
        groupmain = ansmain[2]

    if len(G.nodes()) > MAXSIZE:
        components = list(nx.connected_component_subgraphs(drop(G, k + 1)))
        for component in components:
            g = nx.Graph(component)
            if len(list(g.nodes())) > 1:
                ans = SS(list(g.nodes()))
                scorepred = ans[0]
                spamicity = ans[1]
                group = ans[2]

                if len(list(g.nodes())) > MAXSIZE:
                    FindGroups(g, k + 1)
                elif spamicity >= delta:
                    if len(group.users) > 1:
                        c = 0
                        denom = 0
                        for u in group.users:
                            if u in fake:
                                c = c + 1
                            denom = denom + 1
                        store = (c * 1.0) / denom

                        c = 0
                        denom = 0
                        for u in group.users:
                            for p in group.prods:
                                if p in wholerev[u]:
                                    if int(wholerev[u][p].label) == -1:
                                        c = c + 1
                                    denom = denom + 1

                        index = len(grps) + 1
                        grps[index] = {
                            'id': index,
                            'users': list(group.users),
                            'prods': list(group.prods),
                            'scorepred': scorepred,
                            'scoregt': store,
                            'scoregtreviewprec': (c * 1.0) / denom,
                            'fakegt': spamicity,
                            'fakepred': 0
                        }

                elif len(list(g.nodes())) > 2:
                    FindGroups(g, k + 1)
    elif spamicitymain >= delta:
        if len(groupmain.users) > 1:
            c = 0
            denom = 0
            for u in groupmain.users:
                if u in fake:
                    c = c + 1
                denom = denom + 1
            store = (c * 1.0) / denom

            c = 0
            denom = 0
            for u in groupmain.users:
                for p in groupmain.prods:
                    if p in wholerev[u]:
                        if int(wholerev[u][p].label) == -1:
                            c = c + 1
                        denom = denom + 1

            index = len(grps) + 1
            grps[index] = {
                'id': index,
                'users': list(groupmain.users),
                'prods': list(groupmain.prods),
                'scorepred': scorepredmain,
                'scoregt': store,
                'scoregtreviewprec': (c * 1.0) / denom,
                'fakegt': spamicitymain,
                'fakepred': 0
            }

    else:
        components = list(nx.connected_component_subgraphs(drop(G, k + 1)))
        for component in components:
            g = nx.Graph(component)
            if len(list(g.nodes())) > 1:
                ans = SS(list(g.nodes()))
                scorepred = ans[0]
                spamicity = ans[1]
                group = ans[2]

                if spamicity >= delta:
                    c = 0
                    denom = 0
                    for u in group.users:
                        if u in fake:
                            c = c + 1
                        denom = denom + 1
                    store = (c * 1.0) / denom

                    c = 0
                    denom = 0
                    for u in group.users:
                        for p in group.prods:
                            if p in wholerev[u]:
                                if int(wholerev[u][p].label) == -1:
                                    c = c + 1
                                denom = denom + 1

                    index = len(grps) + 1
                    grps[index] = {
                        'id': index,
                        'users': list(group.users),
                        'prods': list(group.prods),
                        'scorepred': scorepred,
                        'scoregt': store,
                        'scoregtreviewprec': (c * 1.0) / denom,
                        'fakegt': spamicity,
                        'fakepred': 0
                    }

                elif len(list(g.nodes())) > 2:
                    FindGroups(g, k + 1)


grps = {}
conn_comp = list(nx.connected_component_subgraphs(G))
for component in conn_comp:
    G = nx.Graph(component)
    FindGroups(G, 1)

with open('baseline_GSBP.json', 'w') as fp:
    json.dump(grps, fp)
print('end')