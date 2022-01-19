# Focus on a single person in the keypoints files
import json
import os
import sys

target = sys.argv[1]
if not os.path.isdir(target): raise IOError("Must specify a directory of JSON keypoint files.")

def dist_squared(p1, p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2

loc = (0,0)

for filename in sorted(os.listdir(target)):
    all_locs = []
    with open(os.path.join(target+filename), "r") as infile:
        data = json.load(infile)
    for person in data["people"]:
        all_locs.append(person["pose_keypoints_2d"][3:5])
    all_dists = [dist_squared(pos, loc) for pos in all_locs]
    id = all_dists.index(min(all_dists))
    loc = all_locs[id]
    data["people"] = [data["people"][id]]
    with open(os.path.join(target+filename), "w") as outfile:
        json.dump(data, outfile)
