from scipy.spatial import distance
import operator
import json
import numpy as np
import sys
import math
import argparse

parser = argparse.ArgumentParser(description='Ranking groups')
parser.add_argument('--groups', help='path to groups')
parser.add_argument('--ef', help='path to reviewer embeddings')

args = parser.parse_args()


class Groups:
    def __init__(self, users, prods, score, scoregt, id):
        self.users = users
        self.prods = prods
        self.score = score
        self.scoregt = scoregt
        self.id = id

    def __lt__(self, other):
        return len(self.users) < len(other.users)


c = 0
groups = set()
grps = {}
grpmapping = {}
avggrpsize = 0
size = 0
with open(args.groups, 'r') as fp:
    finalgrps = json.load(fp)

filee = open(args.ef, 'r')
mapping = {}
c = 0
for f in filee:
    c = c + 1
    if c == 1:
        continue
    fsplit = f.strip().split(" ")
    if fsplit[0] not in mapping:
        mapping[int(fsplit[0])] = map(float, fsplit[1:])
        emb_size = len(fsplit[1:])

filee.close()

userset = set()
gtscore = {}
size = {}
for g1 in finalgrps:
    finalgrps[g1]['users'] = map(int, finalgrps[g1]['users'])
    finalgrps[g1]['prods'] = map(int, finalgrps[g1]['prods'])

    if len(finalgrps[g1]['users']) > 1:
        scorepred = finalgrps[g1]['scorepred']
        summ = 0
        summ = (sum(scorepred[:5])) / 6.0
        if summ > 0.4:
            group = Groups(finalgrps[g1]['users'], finalgrps[g1]['prods'],
                           finalgrps[g1]['fakegt'], finalgrps[g1]['scoregt'],
                           finalgrps[g1]['id'])
            if len(finalgrps[g1]['users']) not in size:
                size[len(finalgrps[g1]['users'])] = 0
            size[len(finalgrps[g1]['users'])] = size[len(
                finalgrps[g1]['users'])] + 1

            groups.add(group)

            grpmapping[finalgrps[g1]['id']] = group
            if finalgrps[g1]['id'] not in gtscore:
                gtscore[finalgrps[g1]['id']] = finalgrps[g1]['scoregt']

            for u in finalgrps[g1]['users']:
                if u in mapping:
                    userset.add(u)
                if u not in grps:
                    grps[u] = set()
                grps[u].add(finalgrps[g1]['id'])
            avggrpsize = avggrpsize + len(finalgrps[g1]['users'])

r_gt = []
avggrpsize = avggrpsize / (len(groups) * 1.0)
score = {}


def density():
    for gm in grpmapping:
        g = grpmapping[gm]
        avg = [0 for i in range(emb_size)]
        ans = 0
        for u in g.users:
            if u in mapping:
                avg = [avg[i] + mapping[u][i] for i in range(emb_size)]

        avg = [(a * 1.0) / len(g.users) for a in avg]
        for u in g.users:
            if u in mapping:
                ans = ans + distance.euclidean(mapping[u], avg)
        if gm not in score:
            score[gm] = ans / (1.0 * len(g.users))
    sorted_score = sorted(score.items(), key=operator.itemgetter(1))
    return sorted_score


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


def rank():
    sorted_score = density()
    score = {}
    prec = 0
    num = 0
    denom = 0
    f = 0
    c = 1
    precs = []
    prev = 0
    r_baseline = []
    gt_groups = {}
    cc_id = []
    gt_id = []

    for grp in sorted_score:
        if denom == 100:
            break
        denom = denom + 1
        if grp[0] not in gt_groups:
            gt_groups[grp[0]] = grpmapping[grp[0]].scoregt
        r_baseline.append((c, grpmapping[grp[0]].scoregt))
        cc_id.append(grp[0])
        c = c + 1

    r_gt = []
    sorted_score = sorted(gt_groups.items(),
                          key=operator.itemgetter(1),
                          reverse=True)
    c = 0
    for s in sorted_score:
        c = c + 1
        r_gt.append((c, s[1]))
        gt_id.append(s[0])

    dcg(r_baseline, r_gt)
    print '##################'
    inter = 0
    for i in range(0, 100, 10):
        inter = len(set(cc_id[:i + 10]).intersection(set(gt_id[:i + 10])))
        print inter / ((i + 10) * 1.0)


rank()
print 'end'
