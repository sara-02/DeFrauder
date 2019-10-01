import argparse
import json
from scipy.stats import wasserstein_distance
import numpy as np

parser = argparse.ArgumentParser(description='GS RCS groups')
parser.add_argument('--groups', help='path to final groups')
parser.add_argument('--type',
                    help="type gs or rcs to calculate for that feature")

args = parser.parse_args()

finalgrps = {}
avg = 0
with open(args.groups) as fp:
    finalgrps = json.load(fp)

feature = args.type if args.type else 'gs'
print("Calculating: " + feature)
c = {}
co = 0
for j in range(0, 11):
    c[j] = 0
for grp in finalgrps:

    scorepred = finalgrps[grp][feature]
    c[int(scorepred * 10)] = c[int(scorepred * 10)] + 1
    co = co + 1

ans = 0
for v in range(0, 11):
    ans = ans + c[v]
    c[v] = ans

for j in range(0, 11):
    c[j] = c[j] / (co * 1.0)

print(feature, str(wasserstein_distance(np.ones(10), np.array(c.values()))))
print("# groups", str(co))