import networkx as nx
import json
import sys
import time
from datetime import datetime
import operator
import argparse

parser = argparse.ArgumentParser(description='Detecting groups')
parser.add_argument('--metadata', help='path to metadata')
parser.add_argument('--rc', help='path to reviewContent')
parser.add_argument('--dg', help='path to detected groups')

args = parser.parse_args()

CC_mapper = {}


class Review:
    def __init__(self, userid, useridmapped, prodid, prodidmapped, rating,
                 label, date):
        self.userid = userid
        self.useridmapped = useridmapped
        self.prodid = prodid
        self.prodidmapped = prodidmapped
        self.rating = rating
        self.label = label
        self.date = date

    def __repr__(self):
        # Python __repr__() function returns the object representation.
        # It could be any valid python expression such as tuple, dictionary, string etc.
        return '({})'.format(self.userid)

    def __hash__(self):
        return hash((self.userid))

    def __eq__(self, other):
        return self.userid == other.userid

    def __ne__(self, other):
        return not self.__eq__(other)


def isAdjacent(e1, e2):  # QA: Did not understand this function?
    if e1[0] == e2[0] or e1[1] == e2[0] or e1[0] == e2[1] or e1[1] == e2[1]:
        return True
    return False


def degree(
        G, edge
):  # The degree w.r.t two nodes in the edge. -1 to remove repetition.
    return G.degree[edge[0]] + G.degree[edge[1]] - 1


def canbeincluded(userset):
    if len(userset) == 0:
        return 0
    union = set()
    intersection = allusers[list(userset)[0]]
    for u in userset:
        union = union.union(allusers[u])
        intersection = intersection.intersection(allusers[u])
        jaccard = len(intersection) / len(union) * 1.0
        if jaccard > 0.5:
            return 1
        return 0


text = {}
textf = open(args.rc, 'r')
for row in textf:
    # row is of form:
    # userid prodid date review
    # 0 0 2014-02-08 blah-blah-blah-review.
    userid = int(row.split("\t")[1].strip())
    prodid = int(row.split("\t")[0].strip())
    if userid not in text:
        text[userid] = {}
    if prodid not in text[userid]:
        text[userid][prodid] = row.split("\t")[3].strip()
    """
    text is of form:
    {
      0:{
          {
            1: "blah-blah"
          },
          {
            2: "blah-blah"
          }
      },
      9: {
            {
            1: "blah-blah"
          },
          {
            100: "blah-blah"
          }
      }
    }
    userid, and prodid are of type int.
    """

minn = {}
d = {}  # Finally it will store the min_review_date for each prodid(str)
fake = set()  # it will store the userid(str) of fake reviwers
filee = open(args.metadata, 'r')
for f in filee:
    # row is of form:
    # userid prodid rating label date
    # 0      0      5      1   2014-02-08
    fsplit = f.split("\t")
    userid = fsplit[0]
    prodid = fsplit[1]
    rating = int(round(float(fsplit[2])))
    label = fsplit[3]

    if int(label) == -1:
        fake.add(userid)  # The userid in fakeset is of type str.
    """
    For each prodcut we keep track of current review date and min review date.
    """
    # TODO: Redo without minn?
    date = fsplit[4].strip()
    if prodid not in d:  # The prodif in d and minn is of type str.
        minn[prodid] = 0  # QA: Is this line needed?
        d[prodid] = datetime.strptime(date, "%Y-%m-%d").date()

    minn[prodid] = datetime.strptime(date, "%Y-%m-%d").date()
    if minn[prodid] < d[prodid]:
        d[prodid] = minn[prodid]

filee.close()

G = nx.Graph()
reviewsperproddata = {}
nodedetails = {}
prodlist = {}
dictprod = {}
dictprodr = {}
mainnodelist = set()
count = 0
filee = open(args.metadata, 'r')
for f in filee:
    fsplit = f.split("\t")
    userid = fsplit[0]  # type str
    prodid = fsplit[1]  # type str
    rating = str(int(round(float(fsplit[2]))))  # type str
    label = fsplit[3]  # type str
    date = fsplit[4].strip()
    newdate = datetime.strptime(date, "%Y-%m-%d").date()
    datetodays = (newdate - d[prodid]).days #type int
    # #days between first review  ever review of this prodid and this review.
    review = Review(userid, '', prodid, '', rating, label, datetodays)
    # review is presented as
    if prodid + "_" + rating not in reviewsperproddata:
        count = count + 1  # new product-rating combination found.
        reviewsperproddata[prodid + "_" + rating] = set()
        prodlist[prodid + "_" + rating] = []
        dictprod[count] = prodid + "_" + rating
        dictprodr[prodid + "_" + rating] = count
        """
        dictprod: {0:'0_5',1:'0_4'}
        dictprodr: {'0_5':0, '0_4':1}
        """
        # for each unique prod-rating combination add a node in graph.
        G.add_node(count)
    # if prod-rating combo is not new,
    # its count-prod-rating mapping already exists.
    # Save the review obj, in the prodlist.
    prodlist[prodid + "_" + rating].append(review)
    reviewsperproddata[prodid + "_" + rating].add(review)
    """
    prodlist: {'2_3': [(12)], '2_4': [(11)], '1_5': [(5), (6), (8), (9), (10)], '1_3': [(7)]}
    reviewsproddata: {'2_3': set([(12)]), '2_4': set([(11)]),'1_3': set([(7)])}
    i.e for this prod-rating combo which userids reviewed it.
    """

    # Since one user reviews the prod only once,
    # in our case the 2 dicts should be of same length.
    # Also set will be not be able to differentiate 2 reviews as obj ids will be different?
    # print(len(prodlist)==len(reviewsperproddata))
filee.close()
edgedetails = {}
cmnrevrsedges = {}
cmnrevrslist = {}
cmnrevrslistr = {}
cmnrevrsedgeslen = {}
countt = 0
visited = {}
mark = {}
graphlist = list(G.nodes()) # [1,2,3...]
cr = {} # list of reviews who gave p the rating r within time t.
for node1i in range(len(graphlist)):
    # 1st_loop = 0,1,2.... for each node in the graph.
    node1 = graphlist[node1i]
    # node1 = 1, 0th elem in list is 1 and so on.
    if node1 not in cr:
        cr[node1] = []

    for u1i in range(len(prodlist[dictprod[node1]])):
        """
        dictprod[node] -> prod_rating combo.
        prodlist[combo] - > [list of users who reviewed p with rating r.]
        Eg: node1 = 1, cr[1] = [], cr = {1: []}
        2nd_loop = [(5), (6), (8), (9), (10)]
        for each reviwer in the list, check for that reviwer and the rest(a,b)=(b,a)
        """
        u1 = prodlist[dictprod[node1]][u1i]
        # u1i = 0
        # u1 = (5) --> review obj
        cr11 = set()
        cr11.add(u1)
        # cr11 = {5}
        for u2i in range(u1i + 1, len(prodlist[dictprod[node1]])):
            # 3rd_loop = [(6), (8), (9), (10)]
            # u2i = range(1,...)
            # u2i = 1
            u2 = prodlist[dictprod[node1]][u2i]
            # u2 = (6)
            if abs(u1.date - u2.date) < 10: # if the days diff is <10, co-reviwers.
                cr11.add(u2)
            # cr11 = {5,6,8,9,10}
        cr[node1].append(cr11)
        # cr[1] = [{5,6,8,9,10}]
        # in next iteration we review for 6,8,9,10 ; 8,9,10; 9,10
        # cr ={1: [{5,6,9,10},{6,9},{8},{9,10}], 2: [sets()] }
        # No reviewer reviewed around the same time as (8)
    cr[node1].sort(key=len, reverse=True) # largest to smallest.

edgecount = {}
for node1i in range(len(graphlist)):
    node1 = graphlist[node1i]
    for node2i in range(node1i + 1, len(graphlist)):
        node2 = graphlist[node2i]
        maxx = 0
        maxxcr = set()
        cr1 = cr[node1]
        cr2 = cr[node2]
        crlist = set()
        f = 0
        for cri1 in cr1:
            if len(cri1) < 2: # Single nodes. Since cr1 is sorted we can safely exit.
                break # Intersection of single node with any other set does not make sense.
                # If that node had a common time, that would have been captured before.
            for cri2 in cr2:
                if len(cri2) < 2: # Single nodes. Since cr2 is sorted we can safely exit.
                    f = 1
                    break
                crr = cri1.intersection(cri2)
                crr = frozenset(crr) # By taking frozen sets, we reduce duplicacy in crlist.
                if len(crr) > 1:
                # For the case of p1r1 and p1r2, there will be no common nodes.
                    crlist.add(crr)

            if f == 1: # Single nodes. Since cr2 is sorted we can safely exit.
                break

        crlist = list(crlist) # List of frozensets
        crlist.sort(key=len, reverse=True)
        # print(node1)
        # print(node2)
        # print(cr1)
        # print(cr2)
        # print(crlist)
        # sys.exit(1)

        for commonreviewers in crlist:
            if len(commonreviewers) > 1: # redundant check, not needed.

                if commonreviewers not in cmnrevrslistr:
                    countt = countt + 1
                    cmnrevrslist[countt] = commonreviewers
                    cmnrevrslistr[commonreviewers] = countt
                    maincount = countt
                else:
                    maincount = cmnrevrslistr[commonreviewers]
                if node1 < node2:
                    n1 = node1
                    n2 = node2
                else:
                    n1 = node2
                    n2 = node1

                if maincount not in cmnrevrsedges:
                    cmnrevrsedges[maincount] = []

                if (n1, n2) not in edgecount:
                    edgecount[(n1, n2)] = 0
                    G.add_edge(n1, n2)
                    edgedetails[(n1, n2)] = crlist

                if (n1, n2) not in cmnrevrsedges[maincount]:
                    cmnrevrsedges[maincount].append((n1, n2))
                    edgecount[(n1, n2)] = edgecount[(n1, n2)] + 1

for node in G.nodes():
    if G.degree[node] == 0:
        k = frozenset(reviewsperproddata[dictprod[node]])
        if k not in CC_mapper:
            CC_mapper[k] = str(dictprod[node])
        else:
            CC_mapper[k] = CC_mapper[k] + ':' + str(dictprod[node])

poppinglist = []
for item in cmnrevrsedges:
    if len(cmnrevrsedges[item]) == 1:
        poppinglist.append(item)

for p in poppinglist:
    cmnrevrsedges.pop(p)

for c in cmnrevrsedges:
    if c not in cmnrevrsedgeslen:
        mark[c] = 0
        cmnrevrsedgeslen[c] = 0
    cmnrevrsedgeslen[c] = len(cmnrevrsedges[c])
sorted_cmnrevrsedgeslen = sorted(cmnrevrsedgeslen.items(),
                                 key=operator.itemgetter(1))
cmnrevrsedgeslen = sorted_cmnrevrsedgeslen

for ci in range(len(cmnrevrsedgeslen)):
    userset = set()
    prodset = set()
    f = 0
    i = set(cmnrevrslist[cmnrevrsedgeslen[ci][0]])
    for cj in range(ci + 1, len(cmnrevrsedgeslen)):
        j = set(cmnrevrslist[cmnrevrsedgeslen[cj][0]])
        if i.difference(j) == 0:
            if canbeincluded(j):
                mark[cmnrevrsedgeslen[ci][0]] = 1
                mark[cmnrevrsedgeslen[cj][0]] = 1
                userset = userset.union(i.union(j))

                for edge in cmnrevrsedges[cmnrevrsedgeslen[cj][0]]:

                    if cmnrevrslist[cmnrevrsedgeslen[cj][0]] in edgedetails[(
                            edge[0], edge[1])]:
                        edgecount[(
                            edge[0],
                            edge[1])] = edgecount[(edge[0], edge[1])] - 1
                        edgedetails[(edge[0], edge[1])].remove(
                            cmnrevrslist[cmnrevrsedgeslen[cj][0]])
                        if edgecount[(edge[0], edge[1])] == 0:
                            G.remove_edge(edge[0], edge[1])
                        f = 1
            else:
                uset = set()
                pset = set()
                ss = ''
                k = j.difference(i)

                if canbeincluded(k):
                    uset = k
                    k = frozenset(k)
                    pset = allusers[list(uset)[0]]
                    for u in uset:
                        pset = pset.intersection(allusers[u])

                    for p in pset:
                        ss = ss + str(p) + '_0' + ':'

                    ss = ss[0:len(ss) - 1]
                    if k not in CC_mapper:
                        CC_mapper[k] = str(ss)
                    else:
                        CC_mapper[k] = CC_mapper[k] + ':' + ss
                    f = 1

            if f == 1:
                for edge in cmnrevrsedges[cmnrevrsedgeslen[ci][0]]:
                    if cmnrevrslist[cmnrevrsedgeslen[ci][0]] in edgedetails[(
                            edge[0], edge[1])]:
                        edgecount[(
                            edge[0],
                            edge[1])] = edgecount[(edge[0], edge[1])] - 1
                        edgedetails[(edge[0], edge[1])].remove(
                            cmnrevrslist[cmnrevrsedgeslen[ci][0]])
                        if edgecount[(edge[0], edge[1])] == 0:
                            G.remove_edge(edge[0], edge[1])

    ss = ''
    if len(userset) > 0:
        prodset = allusers[list(userset)[0]]
        for u in userset:
            prodset = prodset.intersection(allusers[u])
        for p in prodset:
            ss = ss + str(p) + '_0' + ':'

        ss = ss[0:len(ss) - 1]
        k = frozenset(userset)
        if k not in CC_mapper:
            CC_mapper[k] = str(ss)
        else:
            CC_mapper[k] = CC_mapper[k] + ':' + ss

GG = nx.Graph()
for cmnrvr in cmnrevrsedges:
    nodes = []
    ss = ''
    for edge in cmnrevrsedges[cmnrvr]:
        if edge[0] not in nodes:
            nodes.append(edge[0])
            ss = ss + dictprod[edge[0]] + ":"

        if edge[1] not in nodes:
            nodes.append(edge[1])
            ss = ss + dictprod[edge[1]] + ":"
        edgecount[(edge[0], edge[1])] = edgecount[(edge[0], edge[1])] - 1
        edgedetails[(edge[0], edge[1])].remove(cmnrevrslist[cmnrvr])
        if edgecount[(edge[0], edge[1])] == 0:
            G.remove_edge(edge[0], edge[1])
    ss = ss[0:len(ss) - 1]
    k = frozenset(cmnrevrslist[cmnrvr])
    if k not in CC_mapper:
        CC_mapper[k] = str(ss)
    else:
        CC_mapper[k] = CC_mapper[k] + ':' + ss

co = 0
while len(G.edges()) > 0:

    co = co + 1
    print str(co) + "\t" + str(len(G.edges()))

    cmnrevrsedges2 = {}
    cmnrevrslist2 = {}
    cmnrevrslistr2 = {}
    countt2 = 0
    visited = {}
    edgedetails2 = {}
    edgecount2 = {}
    edgelist = list(G.edges())
    for edge1i in range(len(edgelist)):
        edge1 = edgelist[edge1i]

        if edge1[0] < edge1[1]:
            e10 = edge1[0]
            e11 = edge1[1]
        else:
            e10 = edge1[1]
            e11 = edge1[0]

        s1 = str(dictprod[e10]) + ":" + str(dictprod[e11])

        if edge1i == 0:
            count = count + 1
            node1 = count
            dictprod[count] = s1
            dictprodr[s1] = count
            GG.add_node(count)
            reviewsperproddata[s1] = edgedetails[(e10, e11)]
        else:
            node1 = dictprodr[s1]

        for edge2i in range(edge1i + 1, len(edgelist)):
            edge2 = edgelist[edge2i]

            if edge2[0] < edge2[1]:
                e20 = edge2[0]
                e21 = edge2[1]
            else:
                e20 = edge2[1]
                e21 = edge2[0]

            s2 = str(dictprod[e20]) + ":" + str(dictprod[e21])

            if edge1i == 0:
                count = count + 1
                node2 = count
                dictprod[count] = s2
                dictprodr[s2] = count
                GG.add_node(count)
                reviewsperproddata[s2] = edgedetails[(e20, e21)]
            else:
                node2 = dictprodr[s2]

            if isAdjacent(edge1, edge2):
                cr1 = set(reviewsperproddata[dictprod[node1]])
                cr2 = set(reviewsperproddata[dictprod[node2]])

                crlist = set()
                f = 0
                for cri1 in cr1:
                    if len(cri1) < 2:
                        break
                    for cri2 in cr2:
                        if len(cri2) < 2:
                            f = 1
                            break
                        crr = cri1.intersection(cri2)
                        crr = frozenset(crr)
                        if len(crr) > 1:
                            crlist.add(crr)
                    if f == 1:
                        break

                crlist = list(crlist)
                crlist.sort(key=len, reverse=True)

                for commonreviewers in crlist:

                    if len(commonreviewers
                           ) > 1 and commonreviewers not in cmnrevrslistr:
                        if commonreviewers not in cmnrevrslistr2:
                            countt2 = countt2 + 1
                            cmnrevrslist2[countt2] = commonreviewers
                            cmnrevrslistr2[commonreviewers] = countt2
                            maincount = countt2
                        else:
                            maincount = cmnrevrslistr2[commonreviewers]

                        if maincount not in cmnrevrsedges2:
                            cmnrevrsedges2[maincount] = []

                        if (node1, node2) not in edgecount2:
                            GG.add_edge(node1, node2)
                            edgecount2[(node1, node2)] = 0
                            edgedetails2[(node1, node2)] = crlist
                        if (node1, node2) not in cmnrevrsedges2[maincount]:
                            cmnrevrsedges2[maincount].append((node1, node2))
                            edgecount2[(
                                node1, node2)] = edgecount2[(node1, node2)] + 1

    for node in GG.nodes():
        if GG.degree[node] == 0:

            k = reviewsperproddata[dictprod[node]][0]
            if k not in CC_mapper:
                CC_mapper[k] = str(dictprod[node])
            else:
                CC_mapper[k] = CC_mapper[k] + ':' + str(dictprod[node])

    G = GG
    cmnrevrsedges = cmnrevrsedges2
    cmnrevrslist = cmnrevrslist2
    cmnrevrslistr = cmnrevrslistr2
    edgedetails = edgedetails2
    edgecount = edgecount2
    cmnrevrsedgeslen = {}
    mark = {}

    poppinglist = []
    for item in cmnrevrsedges:
        if len(cmnrevrsedges[item]) == 1:
            poppinglist.append(item)

    for p in poppinglist:
        cmnrevrsedges.pop(p)

    for c in cmnrevrsedges:
        if c not in cmnrevrsedgeslen:
            cmnrevrsedgeslen[c] = 0
            mark[c] = 0
        cmnrevrsedgeslen[c] = len(cmnrevrsedges[c])
    sorted_cmnrevrsedgeslen = sorted(cmnrevrsedgeslen.items(),
                                     key=operator.itemgetter(1))
    cmnrevrsedgeslen = sorted_cmnrevrsedgeslen

    for ci in range(len(cmnrevrsedgeslen)):
        userset = set()
        prodset = set()
        f = 0
        i = set(cmnrevrslist[cmnrevrsedgeslen[ci][0]])
        for cj in range(ci + 1, len(cmnrevrsedgeslen)):
            j = set(cmnrevrslist[cmnrevrsedgeslen[cj][0]])
            if i.difference(j) == 0:
                if canbeincluded(j):
                    mark[cmnrevrsedgeslen[ci][0]] = 1
                    mark[cmnrevrsedgeslen[cj][0]] = 1
                    userset = userset.union(i.union(j))
                    for edge in cmnrevrsedges[cmnrevrsedgeslen[cj][0]]:

                        if cmnrevrslist[cmnrevrsedgeslen[cj]
                                        [0]] in edgedetails[(edge[0],
                                                             edge[1])]:
                            edgecount[(
                                edge[0],
                                edge[1])] = edgecount[(edge[0], edge[1])] - 1
                            edgedetails[(edge[0], edge[1])].remove(
                                cmnrevrslist[cmnrevrsedgeslen[cj][0]])
                            if edgecount[(edge[0], edge[1])] == 0:
                                G.remove_edge(edge[0], edge[1])
                            f = 1
                else:
                    uset = set()
                    pset = set()
                    ss = ''
                    k = j.difference(i)
                    if canbeincluded(k):
                        uset = k
                        k = frozenset(k)
                        pset = allusers[list(uset)[0]]
                        for u in uset:
                            pset = pset.intersection(allusers[u])

                        for p in pset:
                            ss = ss + str(p) + '_0' + ':'

                        ss = ss[0:len(ss) - 1]
                        if k not in CC_mapper:
                            CC_mapper[k] = str(ss)
                        else:
                            CC_mapper[k] = CC_mapper[k] + ':' + ss
                        f = 1
                if f == 1:
                    for edge in cmnrevrsedges[cmnrevrsedgeslen[ci][0]]:
                        if cmnrevrslist[cmnrevrsedgeslen[ci]
                                        [0]] in edgedetails[(edge[0],
                                                             edge[1])]:
                            edgecount[(
                                edge[0],
                                edge[1])] = edgecount[(edge[0], edge[1])] - 1
                            edgedetails[(edge[0], edge[1])].remove(
                                cmnrevrslist[cmnrevrsedgeslen[ci][0]])
                            if edgecount[(edge[0], edge[1])] == 0:
                                G.remove_edge(edge[0], edge[1])

        ss = ''
        if len(userset) > 0:
            prodset = allusers[list(userset)[0]]
            for u in userset:
                prodset = prodset.intersection(allusers[u])
            for p in prodset:
                ss = ss + str(p) + '_0' + ':'

            ss = ss[0:len(ss) - 1]
            k = frozenset(userset)
            if k not in CC_mapper:
                CC_mapper[k] = str(ss)
            else:
                CC_mapper[k] = CC_mapper[k] + ':' + ss

    GG = nx.Graph()

    for cmnrvr in cmnrevrsedges:
        nodes = []
        ss = ''
        for edge in cmnrevrsedges[cmnrvr]:
            if edge[0] not in nodes:
                nodes.append(edge[0])
                ss = ss + dictprod[edge[0]] + ":"

            if edge[1] not in nodes:
                nodes.append(edge[1])
                ss = ss + dictprod[edge[1]] + ":"
            edgecount[(edge[0], edge[1])] = edgecount[(edge[0], edge[1])] - 1
            edgedetails[(edge[0], edge[1])].remove(cmnrevrslist[cmnrvr])
            if edgecount[(edge[0], edge[1])] == 0:
                G.remove_edge(edge[0], edge[1])
        ss = ss[0:len(ss) - 1]
        k = frozenset(cmnrevrslist[cmnrvr])
        if k not in CC_mapper:
            CC_mapper[k] = str(ss)
        else:
            CC_mapper[k] = CC_mapper[k] + ':' + str(ss)

grps = {}
co = 0
for us in CC_mapper:
    c = 0
    denom = 0
    userset = set()
    prodset = set()
    for u in us:
        userset.add(int(u.userid))
    prods = CC_mapper[us].split(':')
    for p in prods:
        prodset.add(int(p.split('_')[0]))

    if len(grps) not in grps and len(prodset) > 0 and len(userset) > 1:
        grps[len(grps)] = {
            'id': len(grps),
            'users': list(userset),
            'prods': list(prodset),
            'scorepred': 0,
            'scoregt': 0,
            'fakegt': 0,
            'fakepred': 0
        }

with open(args.dg, 'w') as fp:
    json.dump(grps, fp)

print 'end'