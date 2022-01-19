import sys
import os
import pickle

target = sys.argv[1]
if not os.path.isdir(target): raise IOError("Must specify a directory of JSON keypoint files.")

keypoint_sequences = []
target_predictions = []

keypoint_sequences_val = []
target_predictions_val = []

file_list = os.listdir(target)
roots = set([filename.split("_")[0] for filename in file_list])

idx_split = int(len(roots)*0.8)
idx = 0

for root in roots:
  idx += 1
  all_keypoints = []
  filenames = [filename for filename in file_list if root in filename]
  for filename in filenames:
    with open(os.path.join(target,filename), "r") as infile:
      data = json.load(infile)
    keypoints = data["people"][0]["pose_keypoints_2d"]
    del keypoints[2::3] # Remove every third element, which contains the confidence for each keypoint, and is not relevant to prediction
    keypoints = keypoints / np.max(np.abs(keypoints),axis=0) # Squeeze everything between 0 and 1
    all_keypoints.append(keypoints)
  if idx < idx_split:
    for i in range(24,len(all_keypoints)-1):
      keypoint_sequences.append(all_keypoints[i-24:i])
      target_predictions.append(all_keypoints[i+1])
  else:
    for i in range(24,len(all_keypoints)-1):
      keypoint_sequences_val.append(all_keypoints[i-24:i])
      target_predictions_val.append(all_keypoints[i+1])

keypoint_sequences = np.array(keypoint_sequences)
target_predictions = np.array(target_predictions)

keypoint_sequences_val = np.array(keypoint_sequences)
target_predictions_val = np.array(target_predictions)

with open("keypoint_sequences.pkl", "wb") as outfile:
  pickle.dump(keypoint_sequences, outfile)
with open("target_predictions.pkl", "wb") as outfile:
  pickle.dump(target_predictions, outfile)
with open("keypoint_sequences_val.pkl", "wb") as outfile:
  pickle.dump(keypoint_sequences_val, outfile)
with open("target_predictions_val.pkl", "wb") as outfile:
  pickle.dump(target_predictions_val, outfile)
