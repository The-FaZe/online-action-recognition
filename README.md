# Online Action Recognition

We built an online action recognition system that recognizes different actions based on UCF-101 Dataset published in 2013. You can refer to it here: https://www.crcv.ucf.edu/data/UCF101.php

Online recognition means that we receive stream of frames from a camera and determine the recognized action directly. That is different from the offline approach which outputs the action after processing a captured video.

Our system is composed of many parts, and is mainly built on top of Temporal Segment Networks.

Temporal Segment Networks for Action Recognition in Videos, Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool, TPAMI, 2018.
[Arxiv Preprint]

# Prerequisites

* The code is written in Python 3.6 with the Anaconda Python Distribution.
* GPUs are required for training.
* Google Colab

We trained and tested our model on two Tesla K-80 GPUs (BA-HPC in Bibliotheca Alexandrina). You can finish training on only one GPU but testing will get you stuck due to the lack of cuda memory as you feed the network one video at a time. 

We ran and debugged all our codes on Google Colab, and once the model is ready for training, we switched to BA_HPC.

# Steps to get the model ready for training (Google Colab)

1. Change Runtime type to choose GPU.
2. Check if the GPU is operating by running the following command:
```
!df ~ --block-size=G
```
if the overall size is more than 300G, you are ready to go.

3. run the following commands to download and extract UCF-101 Dataset:
```
%%sh
# at /content/two-stream-action-recognition
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001 &
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002 &
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003 &
```

```
%%sh
# at /content/two-stream-action-recognition
cat ucf101_jpegs_256.zip.00* > ucf101_jpegs_256.zip &
```

```
# at /content/two-stream-action-recognition
!rm ucf101_jpegs_256.zip.00*
```

```
%%sh
# at /content/two-stream-action-recognition
unzip ucf101_jpegs_256.zip >> zip1.out &
```

```
# at /content/two-stream-action-recognition
!rm ucf101_jpegs_256.zip
```

```
# view the data jpegs_256 should be 33G
!du -sh --human-readable *
```

This should take a while. You can now see the dataset files are ready in the Files section on Google Colab.

4. Use git to clone this repository:
```
!pip install -q xlrd
!git clone https://github.com/The-FaZe/real-time-action-recognition.git
```

# Training

To train our model, use main.py script by running the following:

For RGB model:
```
%%shell

python3 /content/real-time-action-recognition/main.py ucf101 RGB \
  /content/real-time-action-recognition/UCF_lists/rgb_train_FileList1.txt \
  /content/real-time-action-recognition/UCF_lists/rgb_test_FileList1.txt \
   --arch  BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 32 -j 2 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_RGB
```

Hyperparameters tuning was done by the authors of TSN paper and we did a little bit of modifications to suit our GPU capacity. 
You can find what each symbol stands for in the scripting file "parser_commands.py".

For RGB Difference model:
```
%%shell

python3 /content/real-time-action-recognition/main.py ucf101 RGBDiff \
  /content/real-time-action-recognition/UCF_lists/rgb_train_FileList1.txt \
  /content/real-time-action-recognition/UCF_lists/rgb_test_FileList1.txt \
  --arch  BNInception --num_segments 3 \
  --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
  -b 128 -j 8 --dropout 0.8 \
  --gpus 0 1 \
  --KinWeights ~/data/tsn_paper/server_scripts/faze_training/RGB_diff/kinetics_tsn_flow.pth.tar \
  --snapshot_pref ~/data/tsn_paper/server_scripts/faze_training/RGB_diff/UCF101-3seg
```

b & j should be tuned according to your capacity of GPUs and memory size. 
KinWeights refer to the pretrained Kinetcis dataset. In the original work, the model is pretrained on ImageNet dataset to overcome overfitting. Adding Kinetics pretrained weights instead of ImageNet increases accuracy because Kinetics is a large dataset includes 600 different classes for actions.


# Testing








