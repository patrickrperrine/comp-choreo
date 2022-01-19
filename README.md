# Computational Choreography using Human Motion Synthesis
### Patrick Perrine and Trevor Kirkby

## KP-RNN

### Running the Model

### Data Preparation

## Everybody Dance Now

### Setup

#### Run these commands once to set up Everybody Dance Now:
    git clone https://github.com/carolineec/EverybodyDanceNow.git
    cp graph_train.py EverybodyDanceNow/data_prep/graph_train.py
    cp renderopenpose.py EverybodyDanceNow/data_prep/renderopenpose.py
    cp visualizer.py EverybodyDanceNow/util/visualizer.py

#### Install dependencies
    pip install dominate

### Running the Model

Derived from the instructions for [Everybody Dance Now](https://github.com/carolineec/EverybodyDanceNow)

To generate video frames from an already trained EDN model:
```
python EverybodyDanceNow/test_fullts.py \
--name EDN_pretrained_global \
--dataroot \[insert your data folder here\] \
--checkpoints_dir edn_weights \
--results_dir results \
--loadSize 512 \
--no_instance \
--how_many 10000 \
--label_nc 6
```

Or to do the same thing based on weights from TTL data:
```
python EverybodyDanceNow/test_fullts.py \
--name TTL_pretrained_global \
--dataroot \[insert your data folder here\] \
--checkpoints_dir edn_weights \
--results_dir results \
--loadSize 512 \
--no_instance \
--how_many 10000 \
--label_nc 6
```

To convert the frames produced from this into an actual mp4 video:
python resize_frames.py EverybodyDanceNow/results/[model name]/test_latest/images
ffmpeg -i EverybodyDanceNow/results/[model name]/test_latest/images/frame%06d_synthesized_image.png output.mp4

To train the model on new data:
```
python EverybodyDanceNow/train_fullts.py \
--name \[insert name to call this model\] \
--dataroot \[insert your data folder here\] \
--checkpoints_dir checkpoints \
--continue_train \
--loadSize 512 \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6
```

### Data Preparation

This details how one can prepare a folder containing a sequence of images into usable data for the Everybody Dance Now model. The image files should be numbered in ascending order.

In order to prepare new data it is necessary to download and build [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), if you have not already.

OpenPose should be run with the following options:
```
--image_dir [insert your path to the folder of images] --write_json ../EverybodyDanceNow/swing_sample/keypoints --face --hand --display 0 --render_pose 0
```

If the video frames contain multiple people, then run `trim_keypoints.py [insert your path to the folder of JSON keypoints]` to focus on only the pose data from a single person. Be warned that Everybody Dance Now is only designed to work on videos tracking a single individual, but this allows you to experiment with videos containing multiple people.

At this point, make use of one of the data preparation scripts from Everybody Dance Now:
```
python data_prep/graph_avesmooth.py \
--keypoints_dir [insert your path to the folder of JSON keypoints] \
--frames_dir [insert your path to the folder of images] \
--save_dir [insert the path you want the prepared data to be saved to] \
--spread 0 [insert the total number of frames] 1 \
--facetexts
```
