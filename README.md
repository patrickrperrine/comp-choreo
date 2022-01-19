# Computational Choreography using Human Motion Synthesis
## Patrick Perrine and Trevor Kirkby

### Run these commands once to set up Everybody Dance Now:
    git clone https://github.com/carolineec/EverybodyDanceNow.git
    cp graph_train.py EverybodyDanceNow/data_prep/graph_train.py
    cp renderopenpose.py EverybodyDanceNow/data_prep/renderopenpose.py
    cp visualizer.py EverybodyDanceNow/util/visualizer.py

### Install dependencies
    pip install dominate

To generate video from the EDN model:
nohup python EverybodyDanceNow/test_fullts.py \
--name EDN_pretrained_global \
--dataroot subject4/val \
--checkpoints_dir checkpoints \
--results_dir results \
--loadSize 512 \
--no_instance \
--how_many 10000 \
--label_nc 6 &

Or to generate video from the TTL model:
nohup python EverybodyDanceNow/test_fullts.py \
--name TTL_pretrained_global \
--dataroot swing_sample/val \
--checkpoints_dir checkpoints \
--results_dir results \
--loadSize 512 \
--no_instance \
--how_many 10000 \
--label_nc 6 &

Note that for generating a video, the trained model (EDN_pix2pix_global or TTL_pix2pix_global) and the dataset where pose data is being input (subject4/val or swing_sample/val) can also be mixed and matched.

To convert the frames produced from this into an actual mp4 video:
python resize_frames.py EverybodyDanceNow/results/[model name]/test_latest/images
ffmpeg -i EverybodyDanceNow/results/[model name]/test_latest/images/frame%06d_synthesized_image.png output.mp4

To train the model on some of the EDN data:
nohup python EverybodyDanceNow/train_fullts.py \
--name [insert name to call this model] \
--dataroot subject4/train \
--checkpoints_dir checkpoints \
--continue_train \
--loadSize 512 \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6 &

Or to train the model on some of the TTL swing data:
nohup python EverybodyDanceNow/train_fullts.py \
--name [insert name to call this model] \
--dataroot swing_sample/train \
--checkpoints_dir checkpoints \
--continue_train \
--loadSize 512 \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6 &
