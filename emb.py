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

import argparse

parser = argparse.ArgumentParser(description='Creating review2vec')
parser.add_argument('--metadata', help='path to metadata')
parser.add_argument('--rc', help='path to reviewContent')
parser.add_argument('--um', help='path to userIdMapping')
parser.add_argument('--pm', help='path to productIdMapping')
parser.add_argument('--edge', help='path to USERS_edgelistOWN.txt')
parser.add_argument('--adj', help='path to USERS_adjlistOWN.txt')

args = parser.parse_args()


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
userIdMappingfile = open(args.um, 'r')
for row in userIdMappingfile:
    userid = row.split("\t")[1].strip()
    useridmapped = row.split("\t")[0].strip()
    if userid not in userIdMapping:  # QA: Why this if condition?
        userIdMapping[userid] = 0
    userIdMapping[userid] = useridmapped

# print([(k,v) for k,v in userIdMapping.items() if v==0])

productIdMapping = {}
productIdMappingfile = open(args.pm, 'r')
for row in productIdMappingfile:
    prodid = row.split("\t")[1].strip()
    prodidmapped = row.split("\t")[0].strip()
    if prodid not in productIdMapping: # QA: Why this if condition?
        productIdMapping[prodid] = 0 
    productIdMapping[prodid] = prodidmapped

text = []
filee = open(args.rc, 'r')
for f in filee:
    fsplit = f.split("\t")
    text.append(fsplit[3].strip())

filee.close()

allprods = {}
allusers = {}
reviewtime = {}
reviewrating = {}
reviewcontent = {}
wholerev = {}
minn = {}
d = {}
fake = set()
rvdate = {}
maxrvdate = {}
maxrvcon = {}
c = 0
filee = open(args.metadata, 'r')
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
    date = datetime.strptime(date, "%Y-%m-%d").date()
    if prodid not in d:
        minn[prodid] = 0
        d[prodid] = date

    minn[prodid] = date
    if minn[prodid] < d[prodid]:
        d[prodid] = minn[prodid]

    if userid not in rvdate:
        rvdate[userid] = {}
        maxrvdate[userid] = {}
        maxrvcon[userid] = {}
    if prodid not in rvdate[userid]:
        rvdate[userid][prodid] = date
        maxrvdate[userid][prodid] = date
        maxrvcon[userid][prodid] = text[c]

    rvdate[userid][prodid] = date
    if rvdate[userid][prodid] > maxrvdate[userid][prodid]:
        maxrvdate[userid][prodid] = rvdate[userid][prodid]
        maxrvcon[userid][prodid] = text[c]
    c = c + 1

filee.close()

# text=[]
# filee=open("reviewContent",'r')
# for f in filee:
#     fsplit=f.split("\t")
#     text.append(fsplit[3].strip())

# filee.close()

c = 0
filee = open(args.metadata, 'r')
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
    if newdate == maxrvdate[userid][prodid]:

        datetodays = (newdate - d[prodid]).days

        r = Review(userid, useridmapped, prodid, prodidmapped, rating, label,
                   datetodays, maxrvcon[userid][prodid])

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
            reviewcontent[userid][prodid] = maxrvcon[userid][prodid]
        if userid not in allusers:
            allusers[userid] = []
        if prodid not in allprods:
            allprods[prodid] = []
        if userid not in wholerev:
            wholerev[userid] = {}
        if prodid not in wholerev[userid]:
            wholerev[userid][prodid] = r

        allprods[prodid].append(userid)
        allusers[userid].append(prodid)

        c = c + 1

filee.close()
# l=[]
# for a in allprods:
#     l.append(len(allprods[a]))
# l.sort(reverse=True)
# print l[0]
# sys.exit(0)


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

def user_embedding():
    theta = 0.4
    MP = 10000
    tau = 30.0
    suspicion = {}
    adj = {}
    ans = {}
    refinedgroups = []
    fileedgelist = open(args.edge, 'w')
    fileadjlist = open(args.adj, 'w')
    for prod in allprods:
        if prod not in suspicion:
            suspicion[prod] = 1
        suspicion[prod] = (2.0 / (1 + math.exp(
            -1 *
            (pow(abs(MP - len(allprods[prod])), theta)) + pow(2, theta)))) - 1

        userlist = set()
        for u in allprods[prod]:
            userlist.add(u)
            # userlist.add(u.userid)
        userlist = list(userlist)
        for u1i in range(len(userlist)):
            u1 = userlist[u1i]
            if u1 not in ans:
                ans[u1] = {}
            for u2i in range(u1i + 1, len(userlist)):
                u2 = userlist[u2i]
                if u2 not in ans[u1]:
                    ans[u1][u2] = 0
                if abs(reviewtime[u1][prod] - reviewtime[u2][prod]
                       ) <= tau and abs(reviewrating[u1][prod] -
                                        reviewrating[u2][prod]) < 2:
                    ans[u1][u2] = ans[u1][u2] + 0.3 * suspicion[prod] * (
                        (1 -
                         (abs(reviewtime[u1][prod] - reviewtime[u2][prod])) /
                         tau) +
                        (1 -
                         (abs(reviewrating[u1][prod] - reviewrating[u2][prod]))
                         / 2.0) + 0.4 * cosine(reviewcontent[u1][prod],
                                               reviewcontent[u2][prod]))

    for u1 in ans:
        if u1 not in adj:
            adj[u1] = set()
        for u2 in ans[u1]:
            if u2 not in adj:
                adj[u2] = set()
            jaccard = (1.0 * len(
                set(allusers[u1]).intersection(set(allusers[u2])))) / (len(
                    set(allusers[u1]).union(set(allusers[u2]))))
            ans[u1][u2] = ans[u1][u2] * jaccard
            similarity = (2.0 / (1 + math.exp(-1 * ans[u1][u2]))) - 1
            if similarity > 0:
                fileedgelist.write(
                    str(u1) + "\t" + str(u2) + "\t" + str(similarity) + "\n")
                adj[u1].add(u2)
                adj[u2].add(u1)

    for u1 in adj:
        fileadjlist.write(str(u1))
        for u2 in adj[u1]:
            fileadjlist.write("\t" + str(u2))
        fileadjlist.write("\n")

    fileedgelist.close()
    fileadjlist.close()


user_embedding()
print 'end'