# Online Action Recognition

<p align="center">
	<img src="https://github.com/The-FaZe/real-time-action-recognition/blob/master/recognition.PNG" width="350" height="350">
</p>

Kindly, check our latest video [[Youtube Link](https://www.youtube.com/watch?v=-Ztm4shAPHM&t=4s)]

# Table of Contents
* [Introduction](#Introduction)
* [Thesis and Presentation](#Thesis-and-Presentation)
* [Try Our Model](#Try-Our-Model)
* [Prerequisites](#Prerequisites)
* [Dataset Preparation on Google Colab](#Dataset-Preparation-on-Google-Colab)
* [List File Generation](#List-File-Generation)
* [Training](#Training)
* [Testing](#Testing)
* [Contact](#Contact)

----
## Introduction

We built an online action recognition system that recognizes different actions based on UCF-101 Dataset published in 2013. You can refer to it here: https://www.crcv.ucf.edu/data/UCF101.php

Online recognition means that we receive a stream of frames from a camera and determine the recognized action directly. That is different from the offline approach which outputs the action after processing a captured video.

Our system is composed of many parts and is mainly built on top of Temporal Segment Networks.

Temporal Segment Networks for Action Recognition in Videos, Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool, TPAMI, 2018.
[[Arxiv Preprint](https://arxiv.org/abs/1705.02953)]

Our code is well-commented and understandable if you spend some time to keep track of each file and its specific task.

## Thesis and Presentation

You can refer to our thesis book and presentation here for further information.
[[Thesis Book](https://drive.google.com/open?id=1m4O7y54LIowofK2ThnZ2Y6gNKBtwfmVK)]
[[Presentation](https://drive.google.com/file/d/1sCNLpp0VYBiArNoH19p0hyBhD7Kg20Rk/view?usp=sharing)]

## Try Our Model

Note: You can try offline action recognition (ready captured videos), but unfortunately, you cannot try online recognition as you must have access to Bibliotheca Alexandria High Performance Computer(BA-HPC). If you do, please contact one of our members for further help.

1. Open google colab and clone our repos using 
```
!pip install -q xlrd
!git clone https://github.com/The-FaZe/real-time-action-recognition.git
!git clone https://github.com/The-FaZe/Weights.git
```
2. Capture your own video doing any action included in UCF101 dataset.(Capture your video in 480p or 1080p quality)
3. Upload your video to google colab enviroment
4. run Offline_Recognition.py as follows ..
```
%%shell

python3 /content/real-time-action-recognition/Offline_Recognition.py ucf101 \
/content/Weights/RGB_Final.tar \
/content/Weights/RGBDiff_Final.tar \
--arch BNInception --classInd_file /content/real-time-action-recognition/UCF_lists/classInd.txt \
-j 1 --video /content/drive/My\ Drive/Graduation_Project_Team_Memories/Background.mp4 \
--num_segments 3 --sampling_freq 12 --delta 2 --psi 0 --score_weights 1 1.5 --quality 1080p
```
5-You should see .avi file in "/content/real-time-action-recognition" directory, go ahead download and play it.

## Prerequisites

* The code is written in Python 3.6 with the Anaconda Python Distribution.
* GPUs are required for training and testing.
* Google Colab

We trained and tested our model on two Tesla K-80 GPUs (BA-HPC in Bibliotheca Alexandria). You can finish training on only one GPU(eg:Google Colab). but testing will get you stuck due to the lack of Cuda memory as you feed the network one video at a time during test phase. 

We ran and debugged all our codes on Google Colab, and once the model is ready for training, we switched to BA_HPC.

## Dataset Preparation on Google Colab

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
!git clone https://github.com/The-FaZe/Weights.git
```

## List File Generation

You should generate your dataset list file by checking `list_file.py`. You should modify the working directory in the last lines of the code based on your environment.

```
#For any linux machine use..
python3 <List_file directory> <Dataset_file_directory> <Output_directory> <Text_files_directory>

#in case of using google colab

%%shell
python3 /content/real-time-action-recognition/list_file.py \
/content/jpegs_256 /content /content/real-time-action-recognition/UCF_lists

```
Make sure to download UCF_lists folder. then, by running the code, you will have your own train\validation list files generated.

## Training

To train our model, use main.py script by running the following:

For RGB stream:
```
#for any linux machine use .. 

python3 <main.py directroy> ucf101 RGB \
	<train_list_file> <val_list_file> \
   	--arch  BNInception --num_segments 3 \
  	 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
  	 -b 128 -j 8 --dropout 0.8 \
  	 --snapshot_pref <weights_file_name> \
   	 --KinWeights <Kinetics_weights_file_directory>

#for google colab use .. 

%%shell
python3 /content/real-time-action-recognition/main.py ucf101 RGB \
   /content/real-time-action-recognition/UCF_lists/rgb_train_FileList1.txt \
   /content/real-time-action-recognition/UCF_lists/rgb_test_FileList1.txt  \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 128 -j 8  --dropout 0.8 \
   --snapshot_pref Kinetics_BNInception_ \
   --KinWeights /content/Weights/kinetics_tsn_rgb.pth.tar 
  
```

Hyperparameters tuning was done by the authors of TSN paper and we did a little bit of modifications to suit our GPU capacity. 
You can find what each symbol stands for in the scripting file "parser_commands.py".

For RGB Difference stream:
```
#For any linux machine use..
python3 /content/real-time-action-recognition/main.py ucf101 RGBDiff \
<ucf101_rgb_train_list> <ucf101_rgb_val_list> \
  --arch  BNInception --num_segments 3 \
  --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
  -b 128 -j 8 --dropout 0.8 \
  --gpus 0 1 \
  --KinWeights <kinetics_weights_directory> \
  --snapshot_pref <weights_file_name>
 
#For Google Colab use .. 
%%shell
python3 /content/real-time-action-recognition/main.py ucf101 RGBDiff \
   /content/real-time-action-recognition/UCF_lists/rgb_train_FileList1.txt \
   /content/real-time-action-recognition/UCF_lists/rgb_test_FileList1.txt  \
   --arch BNInception --num_segments 3 \
   --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
   -b 64 -j 8  --dropout 0.8 \
   --snapshot_pref Kinetics_BNInception_ \
   --KinWeights /content/Weights/kinetics_tsn_rgb.pth.tar
```

Parameters between <...> should be specified by yourself. 

b & j should be tuned according to your capacity of GPUs and memory size. They should be reduced to suit the GPU in Google Colab.

Note: You can fully train the RGB stream but your GPU memory will fail when training on RGB difference stream as you feed the network with 3 times frames of the RGB stream, so you might need two GPUs not only one.

KinWeights refer to the pretrained Kinetcs dataset. In the original work, the model is pretrained on ImageNet dataset to overcome overfitting. Adding Kinetics pretrained weights instead of ImageNet increases accuracy because Kinetics is a large dataset includes 600 different classes for actions.


## Testing

RGB stream:

```
#For any linux machine use..
python3 <test_models.py directory> ucf101 RGB <ucf101_test_list> <weights_directory> \
	   --gpus 0 1 --arch BNInception --save_scores <score_file_name> \
	   --workers 1 --max_num 5 \
	   --classInd_file <class_index_file>
	      
#For google colab use..
%%shell 
python3 /content/real-time-action-recognition/test_models.py ucf101 RGB \
/content/real-time-action-recognition/UCF_lists/rgb_test_FileList1.txt \
/content/Weights/RGB_Final.tar \
 --gpus 0 --arch BNInception --save_scores rgb_scores --workers 1 --max_num 5 \
 --classInd_file /content/real-time-action-recognition/UCF_lists/classInd.txt
 
```

RGBDiff Stream:
```
#For any linux machine use..
python3 <test_models.py directory> ucf101 RGBDiff <ucf101_test_list> <weights_directory> \
	   --gpus 0 1 --arch BNInception --save_scores <score_file_name> \
	   --workers 1 --max_num 5 \
	   --classInd_file <class_index_file>

#For google colab use..
%%shell 
python3 /content/real-time-action-recognition/test_models.py ucf101 RGBDiff \
/content/real-time-action-recognition/UCF_lists/rgb_test_FileList1.txt \
/content/Weights/RGBDiff_Final.tar \
 --gpus 0 --arch BNInception --save_scores rgbDiff_scores --workers 1 --max_num 5 \
 --classInd_file /content/real-time-action-recognition/UCF_lists/classInd.txt    
```

If you have only one GPU, you can remove gpus & worker parameters.

----
## Contact
For any questions, please contact.
```
Ahmed Gamaleldin: ahmedgamal1496@gmail.com
Ahmed Saied: ahmed1337saied@gmail.com
```
